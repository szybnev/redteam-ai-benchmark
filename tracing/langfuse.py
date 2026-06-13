"""Langfuse tracing integration."""

try:
    from langfuse import Langfuse

    LANGFUSE_AVAILABLE = True
except ImportError:
    Langfuse = None
    LANGFUSE_AVAILABLE = False


class LangfuseTracer:
    """Buffered tracer for Langfuse observability."""

    def __init__(self, config):
        """Initialize Langfuse client."""
        if Langfuse is None:
            raise RuntimeError("Langfuse is not installed")

        self.langfuse = Langfuse(
            public_key=config.public_key,
            secret_key=config.secret_key,
            base_url=config.host,
        )
        self.model_name = None
        self.scoring_method = None
        self.generations = []
        self.optimizations = []
        self.current_optimization = None

    def start_benchmark(self, model_name: str, scoring_method: str) -> None:
        """Start buffered trace for model benchmark."""
        self.model_name = model_name
        self.scoring_method = scoring_method

    def log_generation(
        self,
        question_id: int,
        category: str,
        prompt: str,
        response: str,
        score: int,
        latency_ms: float,
        model: str,
    ) -> None:
        """Buffer LLM generation event."""
        self.generations.append(
            {
                "question_id": question_id,
                "category": category,
                "prompt": prompt,
                "response": response,
                "score": score,
                "latency_ms": latency_ms,
                "model": model,
            }
        )

    def start_optimization(self, question_id: int, category: str) -> None:
        """Start buffered prompt optimization event."""
        self.current_optimization = {
            "question_id": question_id,
            "category": category,
            "attempts": [],
        }

    def log_optimization_attempt(
        self,
        iteration: int,
        strategy: str,
        prompt: str,
        response: str,
        score: int,
        latency_ms: float,
        model: str,
    ) -> None:
        """Buffer optimization iteration event."""
        if self.current_optimization is not None:
            self.current_optimization["attempts"].append(
                {
                    "iteration": iteration,
                    "strategy": strategy,
                    "prompt": prompt,
                    "response": response,
                    "score": score,
                    "latency_ms": latency_ms,
                    "model": model,
                }
            )

    def end_optimization(self, success: bool, best_score: int, iterations: int) -> None:
        """End buffered optimization event."""
        if self.current_optimization is not None:
            self.current_optimization.update(
                {
                    "success": success,
                    "best_score": best_score,
                    "iterations": iterations,
                }
            )
            self.optimizations.append(self.current_optimization)
            self.current_optimization = None

    def end_benchmark(self, total_score: float, interpretation: str) -> None:
        """Flush buffered trace events to Langfuse."""
        if self.current_optimization is not None:
            self.optimizations.append(self.current_optimization)
            self.current_optimization = None

        try:
            trace = self.langfuse.start_span(
                name=f"benchmark-{self.model_name}",
                metadata={
                    "model": self.model_name,
                    "scoring_method": self.scoring_method,
                },
            )

            for event in self.generations:
                gen = trace.start_span(
                    name=f"Q{event['question_id']}-{event['category']}",
                    metadata={
                        "question_id": event["question_id"],
                        "category": event["category"],
                        "score": event["score"],
                        "model": event["model"],
                    },
                )
                gen.update(
                    input=event["prompt"],
                    output=event["response"],
                    usage={"latency_ms": event["latency_ms"]},
                )
                gen.end()

            for optimization in self.optimizations:
                opt_span = trace.start_span(
                    name=f"optimization-Q{optimization['question_id']}",
                    metadata={"category": optimization["category"]},
                )
                for attempt_event in optimization["attempts"]:
                    attempt = opt_span.start_span(
                        name=(
                            f"iter-{attempt_event['iteration']}-"
                            f"{attempt_event['strategy']}"
                        ),
                        metadata={
                            "iteration": attempt_event["iteration"],
                            "strategy": attempt_event["strategy"],
                            "score": attempt_event["score"],
                            "model": attempt_event["model"],
                        },
                    )
                    attempt.update(
                        input=attempt_event["prompt"],
                        output=attempt_event["response"],
                        usage={"latency_ms": attempt_event["latency_ms"]},
                    )
                    attempt.end()
                opt_span.update(
                    metadata={
                        "success": optimization.get("success", False),
                        "best_score": optimization.get("best_score", 0),
                        "iterations": optimization.get("iterations", 0),
                    }
                )
                opt_span.end()

            trace.update(
                metadata={
                    "total_score": total_score,
                    "interpretation": interpretation,
                }
            )
            trace.end()
            self.langfuse.flush()
        except Exception as e:
            print(f"⚠️  Warning: Failed to flush Langfuse trace: {e}")
