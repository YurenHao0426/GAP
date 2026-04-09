"""
Cross-provider model loader implementation.
Allows using different providers for solving and grading tasks.
"""

from typing import Dict, Optional, Tuple, Any
from .base import ModelLoader


class CrossProviderLoader(ModelLoader):
    """Wrapper that allows using different providers for solving and grading."""
    
    def __init__(self, 
                 solver_loader: ModelLoader,
                 grader_loader: Optional[ModelLoader] = None,
                 **kwargs):
        """
        Initialize cross-provider loader.
        
        Args:
            solver_loader: ModelLoader instance for solving problems
            grader_loader: ModelLoader instance for grading (if None, uses solver_loader)
            **kwargs: Additional arguments passed to parent class
        """
        # If no grader loader specified, use the solver loader for both
        self.solver_loader = solver_loader
        self.grader_loader = grader_loader or solver_loader
        
        # Initialize parent with combined model info
        super().__init__(
            solver_model=solver_loader.solver_model,
            grader_model=self.grader_loader.grader_model,
            **kwargs
        )
        
        # Track if we're using cross-provider
        self.is_cross_provider = grader_loader is not None and grader_loader != solver_loader
    
    async def _call_api(self, 
                       model: str, 
                       messages: list[Dict[str, str]], 
                       temperature: float = 0.0) -> Tuple[Optional[str], str]:
        """
        Route API calls to the appropriate provider based on the model.
        
        Args:
            model: Model name to use
            messages: List of messages in chat format
            temperature: Temperature for generation
            
        Returns:
            Tuple of (response_content, raw_response)
        """
        # Determine which loader to use based on the model
        if model == self.solver_model:
            return await self.solver_loader._call_api(model, messages, temperature)
        elif model == self.grader_model:
            return await self.grader_loader._call_api(model, messages, temperature)
        else:
            # Try to determine based on which loader has the model
            if hasattr(self.solver_loader, 'solver_model') and model == self.solver_loader.solver_model:
                return await self.solver_loader._call_api(model, messages, temperature)
            elif hasattr(self.grader_loader, 'grader_model') and model == self.grader_loader.grader_model:
                return await self.grader_loader._call_api(model, messages, temperature)
            else:
                raise ValueError(f"Model {model} not found in either solver or grader loader")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured models and providers."""
        solver_info = self.solver_loader.get_model_info()
        grader_info = self.grader_loader.get_model_info()
        
        return {
            "solver_model": self.solver_model,
            "grader_model": self.grader_model,
            "solver_provider": solver_info.get("provider", "unknown"),
            "grader_provider": grader_info.get("provider", "unknown"),
            "is_cross_provider": self.is_cross_provider,
            "solver_info": solver_info,
            "grader_info": grader_info
        }
    
    async def health_check(self) -> bool:
        """
        Perform health checks on both providers.
        
        Returns:
            True if both providers are healthy, False otherwise
        """
        print("🔍 Checking solver provider health...")
        solver_health = await self.solver_loader.health_check()
        
        if self.is_cross_provider:
            print("🔍 Checking grader provider health...")
            grader_health = await self.grader_loader.health_check()
            return solver_health and grader_health
        else:
            return solver_health
    
    async def estimate_cost(self, 
                          num_problems: int, 
                          avg_problem_length: int = 1000,
                          avg_solution_length: int = 2000) -> Dict[str, float]:
        """
        Estimate costs for both providers.
        
        Args:
            num_problems: Number of problems to process
            avg_problem_length: Average length of problem statements in characters
            avg_solution_length: Average length of solutions in characters
            
        Returns:
            Dictionary with combined cost estimates
        """
        # Get solver costs
        solver_costs = await self.solver_loader.estimate_cost(
            num_problems, avg_problem_length, avg_solution_length
        )
        
        if self.is_cross_provider:
            # Get grader costs separately
            grader_costs = await self.grader_loader.estimate_cost(
                num_problems, avg_problem_length, avg_solution_length
            )
            
            # Combine costs
            return {
                "solver_cost": solver_costs.get("solve_cost", 0),
                "grader_cost": grader_costs.get("grade_cost", 0),
                "total_cost": solver_costs.get("solve_cost", 0) + grader_costs.get("grade_cost", 0),
                "solver_provider": self.solver_loader.get_model_info().get("provider"),
                "grader_provider": self.grader_loader.get_model_info().get("provider"),
                "solver_model": self.solver_model,
                "grader_model": self.grader_model,
                "num_problems": num_problems,
                "note": "Cross-provider costs combined"
            }
        else:
            # Single provider costs
            return solver_costs
    
    async def __aenter__(self):
        """Async context manager entry."""
        if hasattr(self.solver_loader, '__aenter__'):
            await self.solver_loader.__aenter__()
        if self.is_cross_provider and hasattr(self.grader_loader, '__aenter__'):
            await self.grader_loader.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if hasattr(self.solver_loader, '__aexit__'):
            await self.solver_loader.__aexit__(exc_type, exc_val, exc_tb)
        if self.is_cross_provider and hasattr(self.grader_loader, '__aexit__'):
            await self.grader_loader.__aexit__(exc_type, exc_val, exc_tb) 