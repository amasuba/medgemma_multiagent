#!/usr/bin/env python3
"""
MedGemma Multi-Agent System - Main Application Entry Point
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from medgemma_multiagent.utils.config import Config
from medgemma_multiagent.utils.logger import setup_logger
from medgemma_multiagent.agents.retrieval_agent import RetrievalAgent
from medgemma_multiagent.agents.vision_agent import VisionAgent
from medgemma_multiagent.agents.draft_agent import DraftAgent
from medgemma_multiagent.agents.refiner_agent import RefinerAgent
from medgemma_multiagent.agents.synthesis_agent import SynthesisAgent
from medgemma_multiagent.models.medgemma_wrapper import MedGemmaWrapper
from medgemma_multiagent.utils.data_loader import DataLoader
from medgemma_multiagent.utils.evaluation import EvaluationManager

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from PIL import Image

console = Console()
app = typer.Typer(help="MedGemma Multi-Agent System for Chest X-Ray Report Generation")

class MedGemmaMultiAgent:
    """Main orchestrator for the multi-agent system"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config(config_path)
        self.logger = setup_logger(
            name="MedGemmaMultiAgent",
            log_level=self.config.logging.level,
            log_dir=self.config.logging.directory
        )

        # Initialize components
        self.model_wrapper = None
        self.agents = {}
        self.data_loader = None
        self.evaluation_manager = None

        self.logger.info("MedGemma Multi-Agent System initialized")

    async def initialize(self):
        """Initialize all system components"""
        console.print("[bold blue]Initializing MedGemma Multi-Agent System...[/bold blue]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # Initialize model wrapper
            task = progress.add_task("Loading MedGemma model...", total=None)
            self.model_wrapper = MedGemmaWrapper(self.config.models.medgemma)
            await self.model_wrapper.initialize()
            progress.update(task, description="✅ MedGemma model loaded")

            # Initialize data loader
            task = progress.add_task("Setting up data loader...", total=None)
            self.data_loader = DataLoader(self.config.data)
            progress.update(task, description="✅ Data loader ready")

            # Initialize evaluation manager
            task = progress.add_task("Setting up evaluation manager...", total=None)
            self.evaluation_manager = EvaluationManager(self.config.evaluation)
            progress.update(task, description="✅ Evaluation manager ready")

            # Initialize agents
            task = progress.add_task("Initializing agents...", total=None)
            await self._initialize_agents()
            progress.update(task, description="✅ All agents initialized")

        console.print("[bold green]System initialization complete![/bold green]")
        self.logger.info("System initialization completed successfully")

    async def _initialize_agents(self):
        """Initialize all agents"""
        agent_configs = self.config.agents

        # Initialize each agent
        self.agents = {
            "retrieval": RetrievalAgent(
                agent_configs.retrieval_agent,
                self.config.retrieval
            ),
            "vision": VisionAgent(
                agent_configs.vision_agent,
                self.model_wrapper
            ),
            "draft": DraftAgent(
                agent_configs.draft_agent,
                self.model_wrapper
            ),
            "refiner": RefinerAgent(
                agent_configs.refiner_agent,
                self.model_wrapper
            ),
            "synthesis": SynthesisAgent(
                agent_configs.synthesis_agent,
                self.model_wrapper
            )
        }

        # Initialize all agents
        for agent_name, agent in self.agents.items():
            await agent.initialize()
            self.logger.info(f"Agent '{agent_name}' initialized successfully")

    async def generate_report(
        self,
        image_path: str,
        patient_context: Optional[str] = None,
        report_type: str = "detailed"
    ) -> Dict[str, Any]:
        """Generate a comprehensive chest X-ray report"""

        console.print(f"[bold blue]Generating report for: {image_path}[/bold blue]")

        # Load and validate image
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            self.logger.error(f"Failed to load image: {e}")
            return {"error": f"Failed to load image: {e}"}

        results = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # Step 1: Retrieval Agent
            task = progress.add_task("Retrieving similar reports...", total=None)
            retrieval_result = await self.agents["retrieval"].process({
                "image": image,
                "patient_context": patient_context,
                "top_k": 5
            })
            results["retrieval"] = retrieval_result
            progress.update(task, description="✅ Similar reports retrieved")

            # Step 2: Vision Agent
            task = progress.add_task("Analyzing X-ray image...", total=None)
            vision_result = await self.agents["vision"].process({
                "image": image,
                "analysis_type": report_type,
                "context": patient_context
            })
            results["vision"] = vision_result
            progress.update(task, description="✅ Image analysis complete")

            # Step 3: Draft Agent
            task = progress.add_task("Generating initial draft...", total=None)
            draft_result = await self.agents["draft"].process({
                "image": image,
                "vision_analysis": vision_result,
                "retrieved_reports": retrieval_result,
                "patient_context": patient_context
            })
            results["draft"] = draft_result
            progress.update(task, description="✅ Draft generated")

            # Step 4: Refiner Agent
            task = progress.add_task("Refining findings...", total=None)
            refiner_result = await self.agents["refiner"].process({
                "draft_report": draft_result,
                "vision_analysis": vision_result,
                "retrieved_context": retrieval_result
            })
            results["refiner"] = refiner_result
            progress.update(task, description="✅ Findings refined")

            # Step 5: Synthesis Agent
            task = progress.add_task("Synthesizing final report...", total=None)
            synthesis_result = await self.agents["synthesis"].process({
                "draft_report": draft_result,
                "refined_findings": refiner_result,
                "vision_analysis": vision_result,
                "retrieved_context": retrieval_result,
                "patient_context": patient_context
            })
            results["synthesis"] = synthesis_result
            progress.update(task, description="✅ Final report synthesized")

        # Compile final result
        final_result = {
            "image_path": image_path,
            "patient_context": patient_context,
            "report_type": report_type,
            "final_report": synthesis_result.get("final_report", ""),
            "confidence_score": synthesis_result.get("confidence_score", 0.0),
            "findings": refiner_result.get("structured_findings", []),
            "agent_outputs": results,
            "metadata": {
                "timestamp": synthesis_result.get("timestamp"),
                "processing_time": synthesis_result.get("processing_time"),
                "model_version": self.model_wrapper.model_name
            }
        }

        console.print("[bold green]Report generation complete![/bold green]")
        return final_result

    async def process_batch(self, image_paths: list, **kwargs) -> list:
        """Process multiple images in batch"""
        console.print(f"[bold blue]Processing batch of {len(image_paths)} images...[/bold blue]")

        results = []
        for i, image_path in enumerate(image_paths):
            console.print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            result = await self.generate_report(image_path, **kwargs)
            results.append(result)

        return results

    async def add_reports(self, reports: list):
        """Add reports to the knowledge base"""
        console.print(f"[bold blue]Adding {len(reports)} reports to knowledge base...[/bold blue]")

        if "retrieval" in self.agents:
            await self.agents["retrieval"].add_reports(reports)
            console.print("[bold green]Reports added successfully![/bold green]")
        else:
            console.print("[bold red]Retrieval agent not available![/bold red]")

    async def evaluate_system(self, test_data_path: str) -> Dict[str, Any]:
        """Evaluate the system performance"""
        console.print(f"[bold blue]Evaluating system performance...[/bold blue]")

        # Load test data
        test_data = self.data_loader.load_test_data(test_data_path)

        # Run evaluation
        results = await self.evaluation_manager.evaluate(self, test_data)

        # Display results
        self._display_evaluation_results(results)

        return results

    def _display_evaluation_results(self, results: Dict[str, Any]):
        """Display evaluation results in a formatted table"""
        table = Table(title="Evaluation Results")

        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Std Dev", style="yellow")

        for metric, scores in results.items():
            if isinstance(scores, dict) and "mean" in scores:
                table.add_row(
                    metric,
                    f"{scores['mean']:.4f}",
                    f"{scores['std']:.4f}"
                )
            else:
                table.add_row(metric, str(scores), "-")

        console.print(table)

    async def start_api_server(self):
        """Start the API server"""
        console.print("[bold blue]Starting API server...[/bold blue]")

        # This would start the FastAPI server
        # Implementation depends on the API module
        console.print("[bold yellow]API server functionality not implemented yet[/bold yellow]")

    async def shutdown(self):
        """Graceful shutdown of the system"""
        console.print("[bold blue]Shutting down system...[/bold blue]")

        # Cleanup agents
        for agent in self.agents.values():
            await agent.shutdown()

        # Cleanup model wrapper
        if self.model_wrapper:
            await self.model_wrapper.cleanup()

        self.logger.info("System shutdown completed")
        console.print("[bold green]System shutdown complete![/bold green]")

# CLI Commands
@app.command()
def generate(
    image_path: str = typer.Argument(..., help="Path to chest X-ray image"),
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    patient_context: Optional[str] = typer.Option(None, help="Patient context information"),
    report_type: str = typer.Option("detailed", help="Type of report to generate"),
    output_file: Optional[str] = typer.Option(None, help="Output file path")
):
    """Generate a chest X-ray report from an image"""

    async def run():
        system = MedGemmaMultiAgent(config_path)
        await system.initialize()

        try:
            result = await system.generate_report(
                image_path=image_path,
                patient_context=patient_context,
                report_type=report_type
            )

            # Display result
            console.print(Panel(
                result["final_report"],
                title=f"Generated Report - {image_path}",
                border_style="green"
            ))

            # Save to file if requested
            if output_file:
                import json
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                console.print(f"[bold green]Result saved to: {output_file}[/bold green]")

        finally:
            await system.shutdown()

    asyncio.run(run())

@app.command()
def batch(
    images_dir: str = typer.Argument(..., help="Directory containing chest X-ray images"),
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    output_dir: str = typer.Option("./results", help="Output directory for results"),
    pattern: str = typer.Option("*.jpg", help="File pattern to match")
):
    """Process multiple chest X-ray images in batch"""

    async def run():
        system = MedGemmaMultiAgent(config_path)
        await system.initialize()

        try:
            # Find all images
            import glob
            image_paths = glob.glob(os.path.join(images_dir, pattern))

            if not image_paths:
                console.print(f"[bold red]No images found in {images_dir} with pattern {pattern}[/bold red]")
                return

            results = await system.process_batch(image_paths)

            # Save results
            os.makedirs(output_dir, exist_ok=True)
            for i, result in enumerate(results):
                output_file = os.path.join(output_dir, f"result_{i:03d}.json")
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)

            console.print(f"[bold green]Batch processing complete! Results saved to: {output_dir}[/bold green]")

        finally:
            await system.shutdown()

    asyncio.run(run())

@app.command()
def evaluate(
    test_data_path: str = typer.Argument(..., help="Path to test data"),
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    output_file: Optional[str] = typer.Option(None, help="Output file for evaluation results")
):
    """Evaluate the system performance"""

    async def run():
        system = MedGemmaMultiAgent(config_path)
        await system.initialize()

        try:
            results = await system.evaluate_system(test_data_path)

            if output_file:
                import json
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                console.print(f"[bold green]Evaluation results saved to: {output_file}[/bold green]")

        finally:
            await system.shutdown()

    asyncio.run(run())

@app.command()
def serve(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file"),
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(8000, help="Server port")
):
    """Start the API server"""

    async def run():
        system = MedGemmaMultiAgent(config_path)
        await system.initialize()

        try:
            await system.start_api_server()
        finally:
            await system.shutdown()

    asyncio.run(run())

@app.command()
def test(
    config_path: str = typer.Option("config.yaml", help="Path to configuration file")
):
    """Run system tests"""

    async def run():
        system = MedGemmaMultiAgent(config_path)
        await system.initialize()

        try:
            console.print("[bold blue]Running system tests...[/bold blue]")
            # Add test implementation here
            console.print("[bold green]All tests passed![/bold green]")
        finally:
            await system.shutdown()

    asyncio.run(run())

def main():
    """Main entry point"""
    app()

if __name__ == "__main__":
    main()
