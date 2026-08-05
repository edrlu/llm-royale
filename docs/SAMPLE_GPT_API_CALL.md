# Sample GPT API Call

This repo currently sends planner requests to the OpenAI Responses API with:

- `model`: `gpt-5.4-mini` by default
- `temperature`: `0`
- `max_output_tokens`: `220`
- `instructions`: a long system-style prompt
- `input`: a compact JSON string describing the current Clash Royale state

## cURL Example

```bash
curl https://api.openai.com/v1/responses \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4-mini",
    "instructions": "You play Hog 2.6. Return STRICT JSON ONLY, one of: {\"action\":\"idle\",\"reason\":\"...\"} or {\"action\":\"place_card\",\"card\":\"<name>\",\"x_norm\":0.5,\"y_norm\":0.6,\"reason\":\"...\"}. CONTROL LAG: ~1.4s from snapshot to tap. Lead moving targets.",
    "input": "{\"seq\":182,\"w\":720,\"h\":1280,\"decision_latency_sec\":1.4,\"game_phase\":\"1x\",\"board\":{\"river_y_norm\":0.41,\"lane_left_norm\":0.3,\"lane_right_norm\":0.7,\"place_bbox_norm\":{\"x1\":0.02,\"y1\":0.41,\"x2\":0.98,\"y2\":0.77}},\"elixir\":6.0,\"towers\":{\"enemy_left\":62,\"enemy_right\":100,\"friendly_left\":88,\"friendly_right\":100,\"friendly_king\":100,\"enemy_king\":100},\"hand\":[{\"slot\":1,\"label\":\"hog-rider\"},{\"slot\":2,\"label\":\"ice-spirit\"},{\"slot\":3,\"label\":\"cannon\"},{\"slot\":4,\"label\":\"the-log\"}],\"playable_cards\":[\"cannon\",\"hog-rider\",\"ice-spirit\",\"the-log\"],\"playable_with_cost\":[{\"card\":\"cannon\",\"cost\":3},{\"card\":\"hog-rider\",\"cost\":4},{\"card\":\"ice-spirit\",\"cost\":1},{\"card\":\"the-log\",\"cost\":2}],\"cycle\":{\"last_4_played\":[\"skeleton\",\"ice-spirit\"],\"next_cycle_candidates\":[\"musketeer\",\"fireball\"]},\"recent_actions\":[{\"action\":\"idle\",\"t\":112.3}],\"enemy_on_our_side\":[{\"label\":\"hog-rider\",\"lane\":\"left\",\"pressure\":0.93,\"x_norm\":0.32,\"y_norm\":0.57}],\"friendly_on_enemy_side\":[],\"lane_balance\":{\"left\":-1,\"right\":0},\"top_threats\":[{\"label\":\"hog-rider\",\"lane\":\"left\",\"pressure\":0.93,\"x_norm\":0.32,\"y_norm\":0.57}],\"friendly_troops\":[],\"friendly_buildings\":[]}",
    "temperature": 0,
    "max_output_tokens": 220
  }'
```

## Expected Model Output

The code expects JSON text back, for example:

```json
{"action":"place_card","card":"cannon","x_norm":0.50,"y_norm":0.59,"reason":"pull hog to center"}
```

or:

```json
{"action":"idle","reason":"wait for more elixir"}
```

## Python Example

```python
import json
import os
import requests

payload = {
    "model": os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"),
    "instructions": (
        "You play Hog 2.6. Return STRICT JSON ONLY. "
        "CONTROL LAG: ~1.4s from snapshot to tap. Lead moving targets."
    ),
    "input": json.dumps({
        "seq": 182,
        "w": 720,
        "h": 1280,
        "decision_latency_sec": 1.4,
        "game_phase": "1x",
        "elixir": 6.0,
        "playable_cards": ["cannon", "hog-rider", "ice-spirit", "the-log"],
        "enemy_on_our_side": [
            {"label": "hog-rider", "lane": "left", "pressure": 0.93, "x_norm": 0.32, "y_norm": 0.57}
        ],
    }, separators=(",", ":")),
    "temperature": 0,
    "max_output_tokens": 220,
}

response = requests.post(
    "https://api.openai.com/v1/responses",
    headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=60,
)
response.raise_for_status()
data = response.json()

output_text = data.get("output_text", "").strip()
decision = json.loads(output_text)
print(decision)
```

## Notes

- The planner sends `input` as a JSON string, not a nested JSON object.
- Latency is included both in the input as `decision_latency_sec` and in the prompt text as `CONTROL LAG`.
- The live code is implemented in [llm_royale/llm_clasher.py](/Users/edrlu/repos/projects/llm-royale/llm_royale/llm_clasher.py).
