"""
Pipeline script to automatically execute queries from a JSON file.
This script bypasses the GUI and directly uses the ChatService.
Executes each question in BOTH Dense and Sparse modes for comparison.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any

from llama_index.core.llms import ChatMessage, MessageRole

from private_gpt.di import global_injector
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.settings.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryPipeline:
    """
    Pipeline to execute queries automatically from a JSON file.
    Each query is executed in BOTH Dense and Sparse modes for comparison.
    """
    
    def __init__(self):
        """Initialize the pipeline with required services."""
        self.chat_service: ChatService = global_injector.get(ChatService)
        self.ingest_service: IngestService = global_injector.get(IngestService)
        self.system_prompt = settings().ui.default_query_system_prompt
        
    def load_queries(self, queries_file: str = "C:\\Users\\nerea\\Documents\\MasterDTU\\masterThesis\\hbkAgent\\private_gpt\\gt.json") -> list[dict]:
        """
        Load queries from a JSON file.
        
        Expected JSON format (your format):
        [
            {
                "id": "q1",
                "question": "Which solution is suitable for...",
                "reference": "DIRAC Room Acoustics Software...",
                "context": ["...", "..."]
            },
            ...
        ]
        
        Args:
            queries_file: Path to the JSON file containing queries
            
        Returns:
            List of query dictionaries
        """
        queries_path = Path(queries_file)
        
        if not queries_path.exists():
            logger.error(f"Queries file not found: {queries_file}")
            return []
        
        try:
            with open(queries_path, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            logger.info(f"Loaded {len(queries)} queries from {queries_file}")
            return queries
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            return []
    
    def get_context_filter(self, selected_file: str | None) -> ContextFilter | None:
        """
        Create a context filter for a specific file.
        
        Args:
            selected_file: Name of the file to filter by
            
        Returns:
            ContextFilter or None
        """
        if selected_file is None:
            return None
        
        docs_ids = []
        for ingested_document in self.ingest_service.list_ingested():
            if (ingested_document.doc_metadata and 
                ingested_document.doc_metadata.get("file_name") == selected_file):
                docs_ids.append(ingested_document.doc_id)
        
        if docs_ids:
            logger.info(f"Created context filter for file: {selected_file} ({len(docs_ids)} docs)")
            return ContextFilter(docs_ids=docs_ids)
        else:
            logger.warning(f"No documents found for file: {selected_file}")
            return None
    
    def execute_query(
        self, 
        query: str, 
        mode: str = "dense",
        selected_file: str | None = None,
        system_prompt: str | None = None
    ) -> dict[str, Any]:
        """
        Execute a single query and return the response.
        
        Args:
            query: The question to ask
            mode: "dense" or "sparse" RAG mode
            selected_file: Optional file to query against
            system_prompt: Optional custom system prompt
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"  [{mode.upper()}] Executing...")
        
        # Prepare messages
        messages = [ChatMessage(content=query, role=MessageRole.USER)]
        
        # Add system prompt if provided
        if system_prompt or self.system_prompt:
            messages.insert(
                0,
                ChatMessage(
                    content=system_prompt or self.system_prompt,
                    role=MessageRole.SYSTEM
                )
            )
        
        # Get context filter if file is specified
        context_filter = self.get_context_filter(selected_file)
        
        # Determine retriever type
        retriever_type = "bm25" if mode.lower() == "sparse" else "dense"
        
        # Execute query
        try:
            start_time = datetime.now()
            
            completion_gen = self.chat_service.stream_chat(
                messages=messages,
                use_context=True,
                context_filter=context_filter,
                retriever_type=retriever_type
            )
            
            # Collect full response
            full_response = ""
            for delta in completion_gen.response:
                if isinstance(delta, str):
                    full_response += delta
                elif hasattr(delta, 'delta'):
                    full_response += delta.delta or ""
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Extract sources
            sources = []
            if completion_gen.sources:
                for source in completion_gen.sources:
                    doc_metadata = source.document.doc_metadata
                    sources.append({
                        "file": doc_metadata.get("file_name", "-") if doc_metadata else "-",
                        "page": doc_metadata.get("page_label", "-") if doc_metadata else "-",
                        "text_preview": source.text[:200] + "..." if len(source.text) > 200 else source.text
                    })
            
            result = {
                "mode": mode,
                "response": full_response,
                "sources": sources,
                "sources_count": len(sources),
                "execution_time_seconds": round(execution_time, 2),
                "success": True
            }
            
            logger.info(f"  [{mode.upper()}] ✓ Completed in {execution_time:.2f}s | {len(sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"  [{mode.upper()}] ✗ Error: {e}")
            return {
                "mode": mode,
                "response": "",
                "sources": [],
                "sources_count": 0,
                "error": str(e),
                "execution_time_seconds": 0,
                "success": False
            }
    
    def run_pipeline(
        self, 
        queries_file: str = "C:\\Users\\nerea\\Documents\\MasterDTU\\masterThesis\\hbkAgent\\private_gpt\\gt.json",
        output_file: str = "comparison_results.json",
        selected_file: str | None = None
    ) -> None:
        """
        Run the complete pipeline: load queries, execute them in BOTH modes, and save results.
        
        Args:
            queries_file: Path to input JSON with queries
            output_file: Path to output JSON with results
            selected_file: Optional specific file to query against
        """
        logger.info("=" * 80)
        logger.info("Starting Query Comparison Pipeline (Dense vs Sparse)")
        logger.info("=" * 80)
        
        # Load queries
        queries = self.load_queries(queries_file)
        if not queries:
            logger.error("No queries to process. Exiting.")
            return
        
        # Execute queries in BOTH modes
        results = []
        total_queries = len(queries)
        
        for idx, query_data in enumerate(queries, 1):
            logger.info(f"\n{'─' * 80}")
            logger.info(f"[{idx}/{total_queries}] Processing: {query_data.get('id', f'Query {idx}')}")
            logger.info(f"Question: {query_data.get('question', '')[:80]}...")
            logger.info(f"{'─' * 80}")
            
            question = query_data.get("question", "")
            if not question:
                logger.warning(f"Skipping empty question at index {idx}")
                continue
            
            # Execute in DENSE mode
            dense_result = self.execute_query(
                query=question,
                mode="dense",
                selected_file=selected_file,
                system_prompt=None
            )
            
            # Execute in SPARSE mode
            sparse_result = self.execute_query(
                query=question,
                mode="sparse",
                selected_file=selected_file,
                system_prompt=None
            )
            
            # Combine results with original question data
            combined_result = {
                "id": query_data.get("id", f"q{idx}"),
                "question": question,
                "reference": query_data.get("reference", ""),
                "context": query_data.get("context", []),
                "timestamp": datetime.now().isoformat(),
                "dense": dense_result,
                "sparse": sparse_result,
                "comparison": {
                    "dense_sources_count": dense_result.get("sources_count", 0),
                    "sparse_sources_count": sparse_result.get("sources_count", 0),
                    "dense_faster": dense_result.get("execution_time_seconds", 999) < sparse_result.get("execution_time_seconds", 999),
                    "time_difference_seconds": abs(
                        dense_result.get("execution_time_seconds", 0) - 
                        sparse_result.get("execution_time_seconds", 0)
                    )
                }
            }
            
            results.append(combined_result)
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        # Save results
        output_data = {
            "pipeline_info": {
                "execution_date": datetime.now().isoformat(),
                "total_questions": total_queries,
                "successful_questions": len([r for r in results if r["dense"]["success"] and r["sparse"]["success"]]),
                "queries_file": queries_file,
                "selected_file": selected_file
            },
            "summary": summary,
            "results": results
        }
        
        output_path = Path(output_file)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Print summary
            logger.info(f"\n{'=' * 80}")
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"{'=' * 80}")
            logger.info(f"📊 Results saved to: {output_file}")
            logger.info(f"📝 Total questions processed: {total_queries}")
            logger.info(f"✓ Successful comparisons: {output_data['pipeline_info']['successful_questions']}")
            logger.info(f"\n📈 SUMMARY STATISTICS:")
            logger.info(f"  Dense avg time: {summary['dense']['avg_time']:.2f}s")
            logger.info(f"  Sparse avg time: {summary['sparse']['avg_time']:.2f}s")
            logger.info(f"  Dense avg sources: {summary['dense']['avg_sources']:.1f}")
            logger.info(f"  Sparse avg sources: {summary['sparse']['avg_sources']:.1f}")
            logger.info(f"  Dense faster: {summary['comparison']['dense_faster_count']} times")
            logger.info(f"  Sparse faster: {summary['comparison']['sparse_faster_count']} times")
            logger.info(f"{'=' * 80}\n")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _calculate_summary(self, results: list[dict]) -> dict:
        """Calculate summary statistics from results."""
        dense_times = [r["dense"]["execution_time_seconds"] for r in results if r["dense"]["success"]]
        sparse_times = [r["sparse"]["execution_time_seconds"] for r in results if r["sparse"]["success"]]
        dense_sources = [r["dense"]["sources_count"] for r in results if r["dense"]["success"]]
        sparse_sources = [r["sparse"]["sources_count"] for r in results if r["sparse"]["success"]]
        
        dense_faster = sum(1 for r in results if r["comparison"]["dense_faster"])
        sparse_faster = len(results) - dense_faster
        
        return {
            "dense": {
                "avg_time": sum(dense_times) / len(dense_times) if dense_times else 0,
                "min_time": min(dense_times) if dense_times else 0,
                "max_time": max(dense_times) if dense_times else 0,
                "avg_sources": sum(dense_sources) / len(dense_sources) if dense_sources else 0,
            },
            "sparse": {
                "avg_time": sum(sparse_times) / len(sparse_times) if sparse_times else 0,
                "min_time": min(sparse_times) if sparse_times else 0,
                "max_time": max(sparse_times) if sparse_times else 0,
                "avg_sources": sum(sparse_sources) / len(sparse_sources) if sparse_sources else 0,
            },
            "comparison": {
                "dense_faster_count": dense_faster,
                "sparse_faster_count": sparse_faster,
                "total_comparisons": len(results)
            }
        }


def main():
    """Main entry point for the pipeline."""
    pipeline = QueryPipeline()
    
    # Customize these parameters
    pipeline.run_pipeline(
        queries_file="C:\\Users\\nerea\\Documents\\MasterDTU\\masterThesis\\hbkAgent\\private_gpt\\gt.json",  # Your questions file
        output_file="comparison_results.json",  # Output with comparisons
        selected_file=None  # Set to a filename if you want to filter by document
    )


if __name__ == "__main__":
    main()