import json
from pathlib import Path


class JsonlExperimentLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_iteration(
        self,
        iteration,
        attacks,
        responses,
        scores,
        defense_records=None,
    ):
        defense_records = defense_records or [None] * len(responses)
        with self.path.open("a", encoding="utf-8") as handle:
            for stream, (attack, response, score, defense) in enumerate(
                zip(attacks, responses, scores, defense_records), start=1
            ):
                json.dump(
                    {
                        "iteration": iteration,
                        "stream": stream,
                        "improvement": attack.get("improvement"),
                        "prompt": attack.get("prompt"),
                        "released_response": response,
                        "pair_score": score,
                        "defense": defense,
                    },
                    handle,
                    ensure_ascii=False,
                )
                handle.write("\n")
