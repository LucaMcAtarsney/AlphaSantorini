extends Node

signal game_state_received(state: Dictionary)
signal legal_moves_received(payload: Dictionary)
signal request_failed(status_code: int, message: String)

@export var base_url := "http://127.0.0.1:8000"

var _http: HTTPRequest
var _busy := false

var _pending_fetch_after_post := false

func _ready() -> void:
	_http = HTTPRequest.new()
	add_child(_http)
	_http.request_completed.connect(_on_request_completed)

func fetch_game_state() -> void:
	_http_get("%s/game" % base_url)

func fetch_legal_moves() -> void:
	_http_get("%s/game/legal_moves" % base_url)

func step_game() -> void:
	_pending_fetch_after_post = true
	_post_json("%s/game/step" % base_url, {})

func reset_game(payload: Dictionary = {}) -> void:
	_pending_fetch_after_post = true
	_post_json("%s/game/reset" % base_url, payload)

func submit_human_move(payload: Dictionary) -> void:
	_pending_fetch_after_post = false
	_post_json("%s/game/move" % base_url, payload)

func _http_get(url: String) -> void:
	if _busy:
		request_failed.emit(-1, "HTTP busy")
		return
	_busy = true

	var err := _http.request(url)
	if err != OK:
		_busy = false
		request_failed.emit(-1, "HTTPRequest error: %s" % str(err))

func _post_json(url: String, payload: Variant) -> void:
	if _busy:
		request_failed.emit(-1, "HTTP busy")
		return
	_busy = true

	var headers := ["Content-Type: application/json"]
	var body := JSON.stringify(payload)

	var err := _http.request(url, headers, HTTPClient.METHOD_POST, body)
	if err != OK:
		_busy = false
		request_failed.emit(-1, "HTTPRequest error: %s" % str(err))

func _on_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	_busy = false

	if result != HTTPRequest.RESULT_SUCCESS:
		request_failed.emit(-1, "Network error (result=%d)" % result)
		return

	var text := body.get_string_from_utf8()

	if response_code != 200:
		request_failed.emit(response_code, text)
		return

	var parsed = JSON.parse_string(text)
	if parsed == null or typeof(parsed) != TYPE_DICTIONARY:
		request_failed.emit(response_code, "Invalid JSON")
		return

	if parsed.has("board") and parsed.has("workers"):
		_pending_fetch_after_post = false
		game_state_received.emit(_normalize_state(parsed))
		return

	if parsed.has("moves") and parsed.has("current_player"):
		legal_moves_received.emit(_normalize_legal_moves(parsed))
		return

	if _pending_fetch_after_post:
		_pending_fetch_after_post = false
		fetch_game_state()
		return

	request_failed.emit(response_code, "Response was not a game state: %s" % text)

func _normalize_state(state: Dictionary) -> Dictionary:
	var out := state.duplicate(true)

	if out.has("board") and typeof(out["board"]) == TYPE_ARRAY:
		for x in range(out["board"].size()):
			if typeof(out["board"][x]) != TYPE_ARRAY:
				continue
			for y in range(out["board"][x].size()):
				out["board"][x][y] = int(round(float(out["board"][x][y])))

	if out.has("workers") and typeof(out["workers"]) == TYPE_DICTIONARY:
		var w: Dictionary = out["workers"]
		for k in w.keys():
			var arr = w[k]
			if typeof(arr) == TYPE_ARRAY and arr.size() >= 2:
				w[k] = Vector2i(int(round(float(arr[0]))), int(round(float(arr[1]))))

	return out

func _normalize_legal_moves(payload: Dictionary) -> Dictionary:
	var out := payload.duplicate(true)
	if not out.has("moves") or typeof(out["moves"]) != TYPE_ARRAY:
		return out

	for i in range(out["moves"].size()):
		if typeof(out["moves"][i]) != TYPE_DICTIONARY:
			continue
		var move: Dictionary = out["moves"][i]
		for key in ["to", "from", "build"]:
			if move.has(key) and typeof(move[key]) == TYPE_ARRAY and move[key].size() >= 2:
				move[key] = Vector2i(int(round(float(move[key][0]))), int(round(float(move[key][1]))))
		out["moves"][i] = move

	return out
