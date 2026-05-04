extends Node2D

@onready var http: HTTPRequest = $HTTPRequest

const BASE_URL := "http://127.0.0.1:8000"

enum PendingRequest { NONE, GAME, RESET, STEP }
var pending: PendingRequest = PendingRequest.NONE

func _ready() -> void:
	http.request_completed.connect(_on_request_completed)

func _on_game_state_button_up() -> void:
	_request_get(BASE_URL + "/game", PendingRequest.GAME)

func _on_reset_button_up() -> void:
	_request_post_json(BASE_URL + "/game/reset", {}, PendingRequest.RESET)

func _on_step_button_up() -> void:
	_request_post_json(BASE_URL + "/game/step", {}, PendingRequest.STEP)


func _request_get(url: String, tag: PendingRequest) -> void:
	if http.get_http_client_status() != HTTPClient.STATUS_DISCONNECTED:
		print("HTTP busy. Try again in a moment.")
		return

	pending = tag
	print("\n--- GET ", url)
	var err := http.request(url)
	if err != OK:
		print("Request failed to start. Error code: ", err)
		pending = PendingRequest.NONE

func _request_post_json(url: String, payload: Variant, tag: PendingRequest) -> void:
	if http.get_http_client_status() != HTTPClient.STATUS_DISCONNECTED:
		print("HTTP busy. Try again in a moment.")
		return

	pending = tag
	print("\n--- POST ", url)

	var headers := ["Content-Type: application/json"]
	var body := JSON.stringify(payload)

	var err := http.request(url, headers, HTTPClient.METHOD_POST, body)
	if err != OK:
		print("Request failed to start. Error code: ", err)
		pending = PendingRequest.NONE

func _on_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	var label := _pending_label(pending)
	pending = PendingRequest.NONE

	print(label, " HTTP status: ", response_code)

	var text := body.get_string_from_utf8()
	var parsed = JSON.parse_string(text)

	if parsed == null:
		print(label, " Non-JSON response:\n", text)
	else:
		print(label, " JSON response:\n", JSON.stringify(parsed, "\t"))

func _pending_label(tag: PendingRequest) -> String:
	match tag:
		PendingRequest.GAME:  return "[/game]"
		PendingRequest.RESET: return "[/game/reset]"
		PendingRequest.STEP:  return "[/game/step]"
		_: return "[unknown]"
