extends Node3D

const SIZE := 5
const CELL_SPACING := 5.5
const LEVEL_HEIGHT := 0.8
const BLOCK_Y_OFFSETS := [0.0, 0.0, 0.0]
const WORKER_LEVEL_OFFSETS := [0.5, 1.5, 2.5, 2.7, 1.5]
const PLAYER_ONE_NAME := "Agent One"
const PLAYER_TWO_NAME := "Agent Two"
const SPINNER_FRAMES := ["|", "/", "-", "\\"]
const THINKING_LABELS := ["Thinking.", "Thinking..", "Thinking..."]
const THINKING_LABEL_INTERVAL := 0.5
const WORKER_MOVE_ARC_HEIGHT := 1.2
const WORKER_MOVE_HALF_DURATION := 0.18
const WORKER_MOVE_EPSILON := 0.001
const BUILD_REVEAL_DELAY := 0.8
const PLACEMENT_STEP_DELAY := 0.2
const TURN_STEP_DELAY := 2.0
const BUILD_HIGHLIGHT_DURATION := 0.5
const BUILD_TINT_FADE_DURATION := 0.15
const BUILD_TINT_RAMP_RATIO := 0.2
const BUILD_TINT_HOLD_RATIO := 0.25
const BUILD_RISE_DISTANCE := 0.55
const BUILD_START_ALPHA := 0.25
const BUILD_START_TINT_STRENGTH := 0.95
const HUMAN_BUILD_SUBMIT_DELAY := 1.05
const CELL_HIGHLIGHT_ALPHA := 0.22
const CELL_HIGHLIGHT_HEIGHT := 0.08
const CELL_HIGHLIGHT_SIZE_RATIO := 0.72
const CLICK_DRAG_THRESHOLD := 8.0
const INVALID_CELL := Vector2i(-1, -1)
const MOVE_HIGHLIGHT_COLOR := Color8(170, 230, 190)
const BUILD_HIGHLIGHT_COLOR := Color8(145, 220, 175)
const PLACEMENT_HIGHLIGHT_COLOR := Color8(205, 240, 215)
const AGENT_TYPE_VALUES := ["human", "random", "greedy", "heuristic", "minimax-d2", "vanilla-mcts", "mcts-nn", "mcts-heuristic"]
const AGENT_TYPE_LABELS := ["Human", "Random", "Greedy", "Heuristic", "Minimax d2", "Vanilla-MCTS", "MCTS-NN", "MCTS-Heuristic"]

@onready var terrain: Node3D = get_node_or_null("BoardAnchor") as Node3D
@onready var camera: Camera3D = $CameraRig/Pitch/Camera3D
@onready var player_one_type_option: OptionButton = $UI/ModePanel/Margin/VBox/PlayerOneRow/Type
@onready var player_two_type_option: OptionButton = $UI/ModePanel/Margin/VBox/PlayerTwoRow/Type
@onready var player_one_card: Control = $UI/PlayerCards/PlayerOneCard
@onready var player_two_card: Control = $UI/PlayerCards/PlayerTwoCard
@onready var player_one_name_label: Label = $UI/PlayerCards/PlayerOneCard/Margin/VBox/NameRow/Name
@onready var player_two_name_label: Label = $UI/PlayerCards/PlayerTwoCard/Margin/VBox/NameRow/Name
@onready var player_one_spinner: Label = $UI/PlayerCards/PlayerOneCard/Margin/VBox/NameRow/Spinner
@onready var player_two_spinner: Label = $UI/PlayerCards/PlayerTwoCard/Margin/VBox/NameRow/Spinner
@onready var player_one_status: Label = $UI/PlayerCards/PlayerOneCard/Margin/VBox/Status
@onready var player_two_status: Label = $UI/PlayerCards/PlayerTwoCard/Margin/VBox/Status
@onready var play_timer: Timer = $UI/PlayTimer

@export var tile_scene: PackedScene
@export var block_scene1: PackedScene
@export var block_scene2: PackedScene
@export var block_scene3: PackedScene
@export var dome_scene: PackedScene
@export var worker_scene: PackedScene
@export var worker_scene_one: PackedScene
@export var worker_scene_two: PackedScene

const WORKER_COLOR_ONE := Color8(255, 88, 102)
const WORKER_COLOR_TWO := Color8(82, 146, 255)

var cells := []
var cell_highlights := []
var stacks := []
var workers: Dictionary = {}
var worker_positions: Dictionary = {}
var worker_move_tweens: Dictionary = {}
var _active_team := ""
var _spinner_frame := 0
var _thinking_label_frame := 0
var _thinking_label_elapsed := 0.0
var _player_one_name := PLAYER_ONE_NAME
var _player_two_name := PLAYER_TWO_NAME
var _player_one_thinking := false
var _player_two_thinking := false
var _winner_team := ""
var _setting_up_mode_selectors := false
var _pending_board: Array = []
var _build_delay_timer: Timer
var _last_board_state: Array = []
var _pending_build_highlight: Dictionary = {}
var _current_board: Array = []
var _dome_owners: Dictionary = {}
var _legal_moves: Array = []
var _interaction_stage := ""
var _selected_worker := ""
var _selected_move_target := INVALID_CELL
var _preview_worker_key := ""
var _preview_worker_origin_cell := INVALID_CELL
var _preview_worker_origin_position := Vector3.ZERO
var _preview_move_target := INVALID_CELL
var _pending_human_move_payload: Dictionary = {}
var _pointer_down_pos := Vector2.ZERO
var _pointer_click_candidate := false

var _half := (SIZE - 1) * 0.5

const LEVEL_COLORS := [
	Color(0.95, 0.95, 0.95),
	Color(0.7, 0.85, 1.0),
	Color(0.3, 0.55, 0.9),
]

func _ready() -> void:
	if terrain == null:
		terrain = $Terrain
	_build_delay_timer = Timer.new()
	_build_delay_timer.one_shot = true
	_build_delay_timer.wait_time = BUILD_REVEAL_DELAY
	_build_delay_timer.timeout.connect(_on_build_delay_timeout)
	add_child(_build_delay_timer)
	_setup_mode_selectors()
	_setup_player_cards()
	_build_cells()
	Network.game_state_received.connect(apply_state)
	Network.legal_moves_received.connect(_on_legal_moves_received)
	Network.request_failed.connect(_on_network_request_failed)
	_on_reset_button_up()

func _process(delta: float) -> void:
	_spinner_frame = (_spinner_frame + 1) % SPINNER_FRAMES.size()
	_thinking_label_elapsed += delta
	if _thinking_label_elapsed >= THINKING_LABEL_INTERVAL:
		_thinking_label_elapsed = 0.0
		_thinking_label_frame = (_thinking_label_frame + 1) % THINKING_LABELS.size()
	_update_player_cards()

func _build_cells() -> void:
	cells.resize(SIZE)
	cell_highlights.resize(SIZE)
	stacks.resize(SIZE)

	for x in range(SIZE):
		cells[x] = []
		cell_highlights[x] = []
		stacks[x] = []
		cells[x].resize(SIZE)
		cell_highlights[x].resize(SIZE)
		stacks[x].resize(SIZE)

		for y in range(SIZE):
			var cell := Node3D.new()
			cell.name = "Cell_%d_%d" % [x, y]

			var centered_x := (x - _half) * CELL_SPACING
			var centered_z := (y - _half) * CELL_SPACING
			cell.position = Vector3(centered_x, 0.0, centered_z)

			terrain.add_child(cell)
			cells[x][y] = cell

			if tile_scene:
				cell.add_child(tile_scene.instantiate())

			var stack := Node3D.new()
			stack.name = "Stack"
			cell.add_child(stack)
			stacks[x][y] = stack

			var highlight := _create_cell_highlight()
			cell.add_child(highlight)
			cell_highlights[x][y] = highlight

func apply_state(state: Dictionary) -> void:
	if not state.has("board"):
		return
	var board: Array = state["board"]
	var previous_board := _current_board.duplicate(true)
	_sync_dome_owners_with_board(board)
	_sync_preview_with_state(state.get("workers", {}))
	_update_turn_timing(state.get("workers", {}))
	_active_team = _get_active_team(state)
	_winner_team = _detect_winner_team(board, state.get("workers", {}))
	if not _winner_team.is_empty() and playing:
		playing = false
		$UI/HBoxContainer/Play.text = "▶ Play"
		play_timer.stop()
	_apply_player_meta(state)
	var worker_result := _apply_workers(state.get("workers", {}), board)
	_pending_build_highlight = _find_build_highlight(previous_board, board, String(worker_result.get("moved_worker_key", "")))
	_current_board = board.duplicate(true)
	_queue_board_update(board, bool(worker_result.get("moved", false)))
	_update_human_turn_state()
	_update_player_cards()

func _apply_board(board: Array) -> void:
	for x in range(SIZE):
		for y in range(SIZE):
			var stack: Node3D = stacks[x][y]

			for c in stack.get_children():
				c.queue_free()

			var level := 0
			if x < board.size() and y < board[x].size():
				level = int(board[x][y])

			var blocks_to_build = min(level, 3)
			for h in range(blocks_to_build):
				var b
				if h == 0:
					b= block_scene1.instantiate()
				elif h ==1:
					b= block_scene2.instantiate()
				elif h == 2:
					b= block_scene3.instantiate()
				 
				var block_y = LEVEL_HEIGHT * h + LEVEL_HEIGHT * 0.5 + BLOCK_Y_OFFSETS[h]
				b.position = Vector3(0, block_y, 0)
				stack.add_child(b)

			if level >= 4 and dome_scene:
				var dome := dome_scene.instantiate()
				var dome_base_y := 0.0
				if stack.get_child_count() > 0:
					var top_block := stack.get_child(stack.get_child_count() - 1) as Node3D
					if top_block != null:
						dome_base_y = _node_top_y_relative_to_cell(top_block, cells[x][y])
				dome.position = Vector3(0, dome_base_y + 0.5, 0)
				_apply_dome_colour(dome, _dome_owners.get(Vector2i(x, y), ""))
				stack.add_child(dome)


func _apply_workers(w: Dictionary, board: Array) -> Dictionary:
	var worker_moved := false
	var moved_worker_key := ""
	for existing_key in workers.keys():
		var worker_value = w.get(existing_key)
		var has_valid_position := worker_value is Vector2i
		if not w.has(existing_key) or not has_valid_position:
			var node: Node3D = workers[existing_key]
			_kill_worker_tween(existing_key)
			if is_instance_valid(node):
				node.queue_free()
			workers.erase(existing_key)
			worker_positions.erase(existing_key)

	for key in w.keys():
		if not (w[key] is Vector2i):
			continue
		var p: Vector2i = w[key]
		var x := p.x
		var y := p.y

		if not workers.has(key):
			var worker_packed_scene := _get_worker_scene_for_key(key)
			if worker_packed_scene == null:
				continue
			var node := worker_packed_scene.instantiate()
			node.name = "Worker_%s" % str(key)
			terrain.add_child(node)
			workers[key] = node
			
			_apply_worker_colour(node, key)

		var level_here := 0
		if x >= 0 and x < SIZE and y >= 0 and y < SIZE and x < board.size() and y < board[x].size():
			level_here = int(board[x][y])

		var cell_pos = cells[x][y].position
		var wn: Node3D = workers[key]
		var worker_offset_index = clamp(level_here, 0, WORKER_LEVEL_OFFSETS.size() - 1)
		var worker_y = LEVEL_HEIGHT * level_here + WORKER_LEVEL_OFFSETS[worker_offset_index]
		var target_position := Vector3(cell_pos.x, worker_y, cell_pos.z)
		worker_positions[key] = p
		if not worker_move_tweens.has(key) and wn.position == Vector3.ZERO:
			wn.position = target_position
		elif wn.position.distance_to(target_position) > WORKER_MOVE_EPSILON:
			worker_moved = true
			moved_worker_key = key
			_animate_worker_to(key, wn, target_position)

	return {
		"moved": worker_moved,
		"moved_worker_key": moved_worker_key,
	}

func _get_worker_scene_for_key(key: String) -> PackedScene:
	if key.begins_with("ONE_") and worker_scene_one:
		return worker_scene_one
	if key.begins_with("TWO_") and worker_scene_two:
		return worker_scene_two
	return worker_scene

func _apply_worker_colour(worker_root: Node3D, key: String) -> void:
	var color := WORKER_COLOR_TWO
	if key.begins_with("ONE_"):
		color = WORKER_COLOR_ONE
	elif key.begins_with("TWO_"):
		color = WORKER_COLOR_TWO

	var mat := StandardMaterial3D.new()
	mat.albedo_color = color

	var cyl := worker_root.get_node_or_null("CSGCylinder3D") as CSGShape3D
	if cyl:
		cyl.material_override = mat
		return

	for c in worker_root.find_children("*", "CSGShape3D", true, false):
		(c as CSGShape3D).material_override = mat

func _animate_worker_to(key: String, worker: Node3D, target_position: Vector3, on_finished: Callable = Callable()) -> void:
	_kill_worker_tween(key)

	var start_position := worker.position
	var apex_position := start_position.lerp(target_position, 0.5)
	apex_position.y = max(start_position.y, target_position.y) + WORKER_MOVE_ARC_HEIGHT

	var tween := create_tween()
	tween.set_trans(Tween.TRANS_SINE)
	tween.set_ease(Tween.EASE_OUT)
	tween.tween_property(worker, "position", apex_position, WORKER_MOVE_HALF_DURATION)
	tween.set_ease(Tween.EASE_IN)
	tween.tween_property(worker, "position", target_position, WORKER_MOVE_HALF_DURATION)
	tween.finished.connect(func() -> void:
		worker_move_tweens.erase(key)
		if is_instance_valid(worker):
			worker.position = target_position
		if on_finished.is_valid():
			on_finished.call()
	)
	worker_move_tweens[key] = tween

func _kill_worker_tween(key: String) -> void:
	if not worker_move_tweens.has(key):
		return
	var tween: Tween = worker_move_tweens[key]
	if is_instance_valid(tween):
		tween.kill()
	worker_move_tweens.erase(key)

func _queue_board_update(board: Array, delayed: bool) -> void:
	_pending_board = board.duplicate(true)
	if not delayed:
		if not _build_delay_timer.is_stopped():
			_build_delay_timer.stop()
		_apply_pending_board()
		return
	_build_delay_timer.start()

func _apply_pending_board() -> void:
	if _pending_board.is_empty():
		return
	_apply_board(_pending_board)
	_play_pending_build_highlight()
	_last_board_state = _pending_board.duplicate(true)
	_pending_board = []

func _on_build_delay_timeout() -> void:
	_apply_pending_board()

func _find_build_highlight(previous_board: Array, new_board: Array, moved_worker_key: String) -> Dictionary:
	if previous_board.is_empty() or moved_worker_key.is_empty():
		return {}

	for x in range(min(SIZE, new_board.size())):
		if typeof(new_board[x]) != TYPE_ARRAY:
			continue
		for y in range(min(SIZE, new_board[x].size())):
			var previous_height := 0
			if x < previous_board.size() and typeof(previous_board[x]) == TYPE_ARRAY and y < previous_board[x].size():
				previous_height = int(previous_board[x][y])
			var new_height := int(new_board[x][y])
			if new_height > previous_height:
				if new_height >= 4:
					var dome_cell := Vector2i(x, y)
					if not _dome_owners.has(dome_cell):
						_dome_owners[dome_cell] = _team_from_worker_key(moved_worker_key)
				return {
					"x": x,
					"y": y,
					"team": _team_from_worker_key(moved_worker_key),
				}

	return {}

func _play_pending_build_highlight() -> void:
	if _pending_build_highlight.is_empty():
		return

	var x := int(_pending_build_highlight.get("x", -1))
	var y := int(_pending_build_highlight.get("y", -1))
	if x < 0 or x >= SIZE or y < 0 or y >= SIZE:
		_pending_build_highlight = {}
		return

	var stack: Node3D = stacks[x][y]
	if stack.get_child_count() == 0:
		_pending_build_highlight = {}
		return

	var built_node := stack.get_child(stack.get_child_count() - 1) as Node3D
	if built_node == null:
		_pending_build_highlight = {}
		return

	_animate_build_highlight(built_node, _color_for_team(String(_pending_build_highlight.get("team", ""))))
	_pending_build_highlight = {}

func _animate_build_highlight(node: Node3D, team_color: Color) -> void:
	var original_position := node.position
	node.position = original_position - Vector3(0, BUILD_RISE_DISTANCE, 0)

	var animated_materials := _apply_build_highlight_materials(node, team_color)

	var tween := create_tween()
	tween.set_trans(Tween.TRANS_SINE)
	tween.set_ease(Tween.EASE_OUT)
	tween.parallel().tween_property(node, "position", original_position, BUILD_HIGHLIGHT_DURATION)
	for entry in animated_materials:
		var material: StandardMaterial3D = entry["material"]
		var final_color: Color = entry["final_color"]
		var peak_color: Color = entry["peak_color"]
		var ramp_duration := BUILD_TINT_FADE_DURATION * BUILD_TINT_RAMP_RATIO
		var hold_duration := BUILD_TINT_FADE_DURATION * BUILD_TINT_HOLD_RATIO
		var fade_duration = max(BUILD_TINT_FADE_DURATION - ramp_duration - hold_duration, 0.01)
		var color_track := tween.parallel()
		color_track.tween_property(material, "albedo_color", peak_color, ramp_duration)
		color_track.tween_interval(hold_duration)
		color_track.tween_property(material, "albedo_color", final_color, fade_duration)

	tween.finished.connect(func() -> void:
		_clear_build_highlight_materials(animated_materials)
		if is_instance_valid(node):
			node.position = original_position
	)

func _apply_build_highlight_materials(root: Node3D, team_color: Color) -> Array:
	var animated_materials: Array = []

	for shape in root.find_children("*", "CSGShape3D", true, false):
		var csg := shape as CSGShape3D
		var base_color := Color(1, 1, 1, 1)
		var previous_override: Material = csg.material_override
		if csg.material_override is StandardMaterial3D:
			base_color = (csg.material_override as StandardMaterial3D).albedo_color
		var highlight_material := StandardMaterial3D.new()
		highlight_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		highlight_material.albedo_color = _mix_build_color(base_color, team_color, BUILD_START_TINT_STRENGTH)
		highlight_material.albedo_color.a = BUILD_START_ALPHA
		csg.material_override = highlight_material
		animated_materials.append({
			"node": csg,
			"material": highlight_material,
			"previous_override": previous_override,
			"peak_color": Color(highlight_material.albedo_color.r, highlight_material.albedo_color.g, highlight_material.albedo_color.b, 1.0),
			"final_color": Color(base_color.r, base_color.g, base_color.b, 1.0),
		})

	for mesh_node in root.find_children("*", "MeshInstance3D", true, false):
		var mesh := mesh_node as MeshInstance3D
		var base_color := Color(1, 1, 1, 1)
		var previous_override: Material = mesh.material_override
		if mesh.material_override is StandardMaterial3D:
			base_color = (mesh.material_override as StandardMaterial3D).albedo_color
		var highlight_material := StandardMaterial3D.new()
		highlight_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		highlight_material.albedo_color = _mix_build_color(base_color, team_color, BUILD_START_TINT_STRENGTH)
		highlight_material.albedo_color.a = BUILD_START_ALPHA
		mesh.material_override = highlight_material
		animated_materials.append({
			"node": mesh,
			"material": highlight_material,
			"previous_override": previous_override,
			"peak_color": Color(highlight_material.albedo_color.r, highlight_material.albedo_color.g, highlight_material.albedo_color.b, 1.0),
			"final_color": Color(base_color.r, base_color.g, base_color.b, 1.0),
		})

	return animated_materials

func _clear_build_highlight_materials(animated_materials: Array) -> void:
	for entry in animated_materials:
		var visual = entry["node"]
		if is_instance_valid(visual):
			visual.material_override = entry.get("previous_override", null)

func _mix_build_color(base_color: Color, team_color: Color, strength: float) -> Color:
	return Color(
		lerp(base_color.r, team_color.r, strength),
		lerp(base_color.g, team_color.g, strength),
		lerp(base_color.b, team_color.b, strength),
		1.0
	)

func _color_for_team(team: String) -> Color:
	if team == "ONE":
		return WORKER_COLOR_ONE
	if team == "TWO":
		return WORKER_COLOR_TWO
	return Color(1, 1, 1, 1)

func _apply_dome_colour(dome_root: Node3D, team: Variant) -> void:
	var team_text := String(team)
	if team_text.is_empty():
		return
	var color := _color_for_team(team_text)
	_apply_visual_colour(dome_root, color)

	for c in dome_root.find_children("*", "CSGShape3D", true, false):
		_apply_visual_colour(c as Node3D, color)

	for m in dome_root.find_children("*", "MeshInstance3D", true, false):
		_apply_visual_colour(m as Node3D, color)

func _apply_visual_colour(node: Node3D, color: Color) -> void:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	if node is CSGShape3D:
		(node as CSGShape3D).material_override = mat
	elif node is MeshInstance3D:
		(node as MeshInstance3D).material_override = mat

func _team_from_worker_key(worker_key: String) -> String:
	if worker_key.begins_with("ONE_"):
		return "ONE"
	if worker_key.begins_with("TWO_"):
		return "TWO"
	return ""

func _setup_player_cards() -> void:
	_update_player_cards()

func _setup_mode_selectors() -> void:
	_setting_up_mode_selectors = true
	player_one_type_option.clear()
	player_two_type_option.clear()
	for i in range(AGENT_TYPE_VALUES.size()):
		player_one_type_option.add_item(AGENT_TYPE_LABELS[i], i)
		player_two_type_option.add_item(AGENT_TYPE_LABELS[i], i)
	player_one_type_option.select(0)
	player_two_type_option.select(2)
	player_one_type_option.item_selected.connect(_on_player_type_selected)
	player_two_type_option.item_selected.connect(_on_player_type_selected)
	_setting_up_mode_selectors = false

func _on_player_type_selected(_index: int) -> void:
	if _setting_up_mode_selectors:
		return
	_on_reset_button_up()

func _update_player_cards() -> void:
	var player_one_active := _active_team == "ONE"
	var player_two_active := _active_team == "TWO"
	player_one_name_label.text = _player_one_name
	player_two_name_label.text = _player_two_name

	player_one_spinner.visible = _player_one_thinking and _winner_team.is_empty()
	player_two_spinner.visible = _player_two_thinking and _winner_team.is_empty()
	player_one_spinner.text = SPINNER_FRAMES[_spinner_frame]
	player_two_spinner.text = SPINNER_FRAMES[_spinner_frame]

	player_one_status.text = _get_player_status("ONE", _player_one_thinking, player_one_active)
	player_two_status.text = _get_player_status("TWO", _player_two_thinking, player_two_active)

	player_one_card.modulate = Color(1, 1, 1, 1) if player_one_active else Color(0.92, 0.92, 0.96, 0.95)
	player_two_card.modulate = Color(1, 1, 1, 1) if player_two_active else Color(0.92, 0.92, 0.96, 0.95)

func _apply_player_meta(state: Dictionary) -> void:
	_player_one_name = PLAYER_ONE_NAME
	_player_two_name = PLAYER_TWO_NAME
	_player_one_thinking = false
	_player_two_thinking = false

	if not state.has("players") or typeof(state["players"]) != TYPE_DICTIONARY:
		return

	var players: Dictionary = state["players"]
	if players.has("ONE") and typeof(players["ONE"]) == TYPE_DICTIONARY:
		var one: Dictionary = players["ONE"]
		_player_one_name = str(one.get("name", PLAYER_ONE_NAME))
		_player_one_thinking = bool(one.get("is_thinking", false))

	if players.has("TWO") and typeof(players["TWO"]) == TYPE_DICTIONARY:
		var two: Dictionary = players["TWO"]
		_player_two_name = str(two.get("name", PLAYER_TWO_NAME))
		_player_two_thinking = bool(two.get("is_thinking", false))

func _get_player_status(team: String, is_thinking: bool, is_active: bool) -> String:
	if _winner_team == team:
		return "Winner"
	if not _winner_team.is_empty():
		return ":("
	if not playing and is_active:
		return "Your turn"
	if is_thinking or is_active:
		return THINKING_LABELS[_thinking_label_frame]
	return "Waiting"

func _detect_winner_team(board: Array, worker_state: Variant) -> String:
	if typeof(worker_state) != TYPE_DICTIONARY:
		return ""
	for key in (worker_state as Dictionary).keys():
		var cell = worker_state[key]
		if not (cell is Vector2i):
			continue
		var board_cell := cell as Vector2i
		if board_cell.x < 0 or board_cell.x >= board.size():
			continue
		if typeof(board[board_cell.x]) != TYPE_ARRAY or board_cell.y < 0 or board_cell.y >= board[board_cell.x].size():
			continue
		if int(board[board_cell.x][board_cell.y]) >= 3:
			return _team_from_worker_key(String(key))
	return ""

func _update_turn_timing(worker_state: Variant) -> void:
	if play_timer == null:
		return
	play_timer.wait_time = PLACEMENT_STEP_DELAY if _is_placement_phase(worker_state) else TURN_STEP_DELAY

func _is_placement_phase(worker_state: Variant) -> bool:
	if typeof(worker_state) != TYPE_DICTIONARY:
		return false
	var placed_workers := 0
	for value in (worker_state as Dictionary).values():
		if value is Vector2i:
			placed_workers += 1
	return placed_workers < 4

func _get_active_team(state: Dictionary) -> String:
	for key in ["current_player", "current_turn", "turn", "active_player", "active_agent", "agent", "player"]:
		if state.has(key):
			return _normalize_team_value(state[key])
	return ""

func _normalize_team_value(value: Variant) -> String:
	if typeof(value) == TYPE_INT:
		if int(value) == 1:
			return "ONE"
		if int(value) == 2:
			return "TWO"
	if typeof(value) == TYPE_STRING:
		var text := String(value).to_upper()
		if "ONE" in text or text == "1":
			return "ONE"
		if "TWO" in text or text == "2":
			return "TWO"
	return ""

func _create_cell_highlight() -> MeshInstance3D:
	var highlight := MeshInstance3D.new()
	highlight.name = "CellHighlight"
	var mesh := BoxMesh.new()
	mesh.size = Vector3(CELL_SPACING * 0.72, 0.06, CELL_SPACING * 0.72)
	highlight.mesh = mesh
	var material := StandardMaterial3D.new()
	material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material.albedo_color = Color(1, 1, 1, CELL_HIGHLIGHT_ALPHA)
	material.emission_enabled = true
	material.emission = Color(1, 1, 1)
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	highlight.material_override = material
	highlight.visible = false
	return highlight

func _update_human_turn_state() -> void:
	if not _winner_team.is_empty():
		_clear_human_interaction()
		return
	if _is_human_turn():
		if playing:
			playing = false
			$UI/HBoxContainer/Play.text = "▶ Play"
			play_timer.stop()
		Network.fetch_legal_moves()
	else:
		_clear_human_interaction()

func _is_human_turn() -> bool:
	return _is_human_team(_active_team)

func _is_human_team(team: String) -> bool:
	if team == "ONE":
		return _player_one_name.to_lower().begins_with("player ")
	if team == "TWO":
		return _player_two_name.to_lower().begins_with("player ")
	return false

func _on_legal_moves_received(payload: Dictionary) -> void:
	if _normalize_team_value(payload.get("current_player", "")) != _active_team or not _is_human_turn():
		return
	_legal_moves = payload.get("moves", [])
	_selected_worker = ""
	_selected_move_target = INVALID_CELL
	if _has_placement_moves():
		_interaction_stage = "placement"
	elif _legal_moves.is_empty():
		_interaction_stage = ""
	else:
		_interaction_stage = "select_worker"
	_refresh_human_highlights()

func _on_network_request_failed(status_code: int, message: String) -> void:
	push_warning("Network request failed (%d): %s" % [status_code, message])

func _clear_human_interaction() -> void:
	_legal_moves = []
	_interaction_stage = ""
	_selected_worker = ""
	_selected_move_target = INVALID_CELL
	if not _preview_worker_key.is_empty():
		_reset_worker_preview()
	_refresh_human_highlights()

func _has_placement_moves() -> bool:
	for move in _legal_moves:
		if typeof(move) == TYPE_DICTIONARY and move.get("type", "") == "placement":
			return true
	return false

func _refresh_human_highlights() -> void:
	_clear_cell_highlights()
	_reset_worker_highlights()

	if not _is_human_turn():
		return

	if _interaction_stage == "placement":
		for move in _legal_moves:
			if typeof(move) != TYPE_DICTIONARY or move.get("type", "") != "placement":
				continue
			_show_cell_highlight(move["to"], PLACEMENT_HIGHLIGHT_COLOR, _current_cell_surface_height(move["to"], false))
		return

	var worker_ids := _legal_worker_ids()
	for worker_id in worker_ids:
		_set_worker_highlight(worker_id, worker_id == _selected_worker)

	if _interaction_stage == "move_select":
		for target in _legal_move_targets_for_worker(_selected_worker):
			_show_cell_highlight(target, MOVE_HIGHLIGHT_COLOR, _current_cell_surface_height(target, false))
	elif _interaction_stage == "build_select":
		for target in _legal_build_targets_for_selection():
			_show_cell_highlight(target, BUILD_HIGHLIGHT_COLOR, _current_cell_surface_height(target, true))

func _clear_cell_highlights() -> void:
	for x in range(SIZE):
		for y in range(SIZE):
			var highlight: MeshInstance3D = cell_highlights[x][y]
			if highlight:
				highlight.visible = false

func _show_cell_highlight(cell: Vector2i, color: Color, y_height: float) -> void:
	if cell.x < 0 or cell.x >= SIZE or cell.y < 0 or cell.y >= SIZE:
		return
	var highlight: MeshInstance3D = cell_highlights[cell.x][cell.y]
	if highlight == null:
		return
	highlight.position = Vector3(0, y_height + CELL_HIGHLIGHT_HEIGHT, 0)
	var material := highlight.material_override as StandardMaterial3D
	material.albedo_color = Color(color.r, color.g, color.b, CELL_HIGHLIGHT_ALPHA)
	material.emission = color
	highlight.visible = true

func _reset_worker_highlights() -> void:
	for key in workers.keys():
		var worker: Node3D = workers[key]
		if is_instance_valid(worker):
			worker.scale = Vector3.ONE

func _set_worker_highlight(worker_key: String, selected: bool) -> void:
	if not workers.has(worker_key):
		return
	var worker: Node3D = workers[worker_key]
	if not is_instance_valid(worker):
		return
	worker.scale = Vector3.ONE * (1.18 if selected else 1.08)

func _legal_worker_ids() -> Array:
	var ids: Array = []
	for move in _legal_moves:
		if typeof(move) != TYPE_DICTIONARY or move.get("type", "") != "turn":
			continue
		var worker_id := String(move.get("worker", ""))
		if not ids.has(worker_id):
			ids.append(worker_id)
	return ids

func _legal_move_targets_for_worker(worker_key: String) -> Array:
	var targets: Array = []
	for move in _legal_moves:
		if typeof(move) != TYPE_DICTIONARY:
			continue
		if move.get("type", "") != "turn" or String(move.get("worker", "")) != worker_key:
			continue
		var target: Vector2i = move.get("to", INVALID_CELL)
		if target != INVALID_CELL and not targets.has(target):
			targets.append(target)
	return targets

func _legal_build_targets_for_selection() -> Array:
	var targets: Array = []
	for move in _legal_moves:
		if typeof(move) != TYPE_DICTIONARY:
			continue
		if move.get("type", "") != "turn":
			continue
		if String(move.get("worker", "")) != _selected_worker:
			continue
		if move.get("to", INVALID_CELL) != _selected_move_target:
			continue
		var target: Vector2i = move.get("build", INVALID_CELL)
		if target != INVALID_CELL and not targets.has(target):
			targets.append(target)
	return targets

func _current_cell_surface_height(cell: Vector2i, build_preview: bool) -> float:
	if cell.x < 0 or cell.x >= SIZE or cell.y < 0 or cell.y >= SIZE:
		return 0.0
	var stack: Node3D = stacks[cell.x][cell.y]
	if stack == null or stack.get_child_count() == 0:
		return 0.0
	var cell_root: Node3D = cells[cell.x][cell.y]
	var surface_y := 0.0

	for child in stack.get_children():
		if child is Node3D:
			surface_y = max(surface_y, _node_top_y_relative_to_cell(child as Node3D, cell_root))

	return surface_y

func _stack_top_y_relative_to_cell(stack: Node3D, cell_root: Node3D) -> float:
	var surface_y := 0.0
	for child in stack.get_children():
		if child is Node3D:
			surface_y = max(surface_y, _node_top_y_relative_to_cell(child as Node3D, cell_root))
	return surface_y

func _node_top_y_relative_to_cell(root: Node3D, cell_root: Node3D) -> float:
	var top_y := 0.0

	if root is VisualInstance3D and root.has_method("get_aabb"):
		top_y = max(top_y, _aabb_top_y_in_cell_space(root as VisualInstance3D, cell_root))

	for child in root.get_children():
		if child is Node3D:
			top_y = max(top_y, _node_top_y_relative_to_cell(child as Node3D, cell_root))

	return top_y

func _aabb_top_y_in_cell_space(visual: VisualInstance3D, cell_root: Node3D) -> float:
	var aabb: AABB = visual.get_aabb()
	var top_y := -INF

	for corner in [
		Vector3(aabb.position.x, aabb.position.y, aabb.position.z),
		Vector3(aabb.position.x + aabb.size.x, aabb.position.y, aabb.position.z),
		Vector3(aabb.position.x, aabb.position.y + aabb.size.y, aabb.position.z),
		Vector3(aabb.position.x, aabb.position.y, aabb.position.z + aabb.size.z),
		Vector3(aabb.position.x + aabb.size.x, aabb.position.y + aabb.size.y, aabb.position.z),
		Vector3(aabb.position.x + aabb.size.x, aabb.position.y, aabb.position.z + aabb.size.z),
		Vector3(aabb.position.x, aabb.position.y + aabb.size.y, aabb.position.z + aabb.size.z),
		Vector3(aabb.position.x + aabb.size.x, aabb.position.y + aabb.size.y, aabb.position.z + aabb.size.z),
	]:
		var world_corner := visual.to_global(corner)
		var local_corner := cell_root.to_local(world_corner)
		top_y = max(top_y, local_corner.y)

	return 0.0 if top_y == -INF else top_y

func _worker_at_cell(cell: Vector2i) -> String:
	for key in worker_positions.keys():
		if worker_positions[key] == cell:
			return String(key)
	return ""

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
		if event.pressed:
			_pointer_down_pos = event.position
			_pointer_click_candidate = true
		else:
			if _pointer_click_candidate and _is_human_turn():
				var cell := _pick_cell_from_screen(event.position)
				if cell != INVALID_CELL:
					_handle_human_cell_click(cell)
			_pointer_click_candidate = false
	elif event is InputEventMouseMotion and _pointer_click_candidate:
		if event.position.distance_to(_pointer_down_pos) > CLICK_DRAG_THRESHOLD:
			_pointer_click_candidate = false

func _pick_cell_from_screen(screen_pos: Vector2) -> Vector2i:
	if camera == null or terrain == null:
		return INVALID_CELL
	var origin := camera.project_ray_origin(screen_pos)
	var direction := camera.project_ray_normal(screen_pos)
	var highlighted_hit := _pick_visible_highlight(origin, direction)
	if highlighted_hit != INVALID_CELL:
		return highlighted_hit
	var stack_hit := _pick_stack_geometry(origin, direction)
	if stack_hit != INVALID_CELL:
		return stack_hit
	if abs(direction.y) < 0.0001:
		return INVALID_CELL
	var t := (terrain.global_position.y - origin.y) / direction.y
	if t < 0.0:
		return INVALID_CELL
	var world_hit := origin + direction * t
	var local_hit := terrain.to_local(world_hit)
	var nearest := INVALID_CELL
	var best_distance := INF
	for x in range(SIZE):
		for y in range(SIZE):
			var distance := Vector2(local_hit.x, local_hit.z).distance_to(Vector2(cells[x][y].position.x, cells[x][y].position.z))
			if distance < best_distance:
				best_distance = distance
				nearest = Vector2i(x, y)
	if best_distance > CELL_SPACING * 0.7:
		return INVALID_CELL
	return nearest

func _pick_visible_highlight(origin: Vector3, direction: Vector3) -> Vector2i:
	if abs(direction.y) < 0.0001:
		return INVALID_CELL

	var best_t := INF
	var best_cell := INVALID_CELL
	var half_extent := CELL_SPACING * CELL_HIGHLIGHT_SIZE_RATIO * 0.5

	for x in range(SIZE):
		for y in range(SIZE):
			var highlight: MeshInstance3D = cell_highlights[x][y]
			if highlight == null or not highlight.visible:
				continue

			var plane_y := highlight.global_position.y
			var t := (plane_y - origin.y) / direction.y
			if t < 0.0 or t >= best_t:
				continue

			var hit := origin + direction * t
			var local_hit = cells[x][y].to_local(hit)
			if abs(local_hit.x) <= half_extent and abs(local_hit.z) <= half_extent:
				best_t = t
				best_cell = Vector2i(x, y)

	return best_cell

func _pick_stack_geometry(origin: Vector3, direction: Vector3) -> Vector2i:
	var best_t := INF
	var best_cell := INVALID_CELL

	for x in range(SIZE):
		for y in range(SIZE):
			var stack: Node3D = stacks[x][y]
			if stack == null or stack.get_child_count() == 0:
				continue
			for child in stack.get_children():
				if child is Node3D:
					var hit_t := _ray_hits_node_geometry(origin, direction, child as Node3D)
					if hit_t >= 0.0 and hit_t < best_t:
						best_t = hit_t
						best_cell = Vector2i(x, y)

	return best_cell

func _ray_hits_node_geometry(origin: Vector3, direction: Vector3, root: Node3D) -> float:
	var best_t := INF

	if root is VisualInstance3D and root.has_method("get_aabb"):
		var hit_t := _ray_hits_visual_aabb(origin, direction, root as VisualInstance3D)
		if hit_t >= 0.0:
			best_t = min(best_t, hit_t)

	for child in root.get_children():
		if child is Node3D:
			var child_t := _ray_hits_node_geometry(origin, direction, child as Node3D)
			if child_t >= 0.0:
				best_t = min(best_t, child_t)

	return -1.0 if best_t == INF else best_t

func _ray_hits_visual_aabb(origin: Vector3, direction: Vector3, visual: VisualInstance3D) -> float:
	var world_aabb := _world_aabb_for_visual(visual)
	return _ray_intersects_aabb(origin, direction, world_aabb)

func _world_aabb_for_visual(visual: VisualInstance3D) -> AABB:
	var aabb: AABB = visual.get_aabb()
	var min_corner := Vector3(INF, INF, INF)
	var max_corner := Vector3(-INF, -INF, -INF)

	for corner in [
		Vector3(aabb.position.x, aabb.position.y, aabb.position.z),
		Vector3(aabb.position.x + aabb.size.x, aabb.position.y, aabb.position.z),
		Vector3(aabb.position.x, aabb.position.y + aabb.size.y, aabb.position.z),
		Vector3(aabb.position.x, aabb.position.y, aabb.position.z + aabb.size.z),
		Vector3(aabb.position.x + aabb.size.x, aabb.position.y + aabb.size.y, aabb.position.z),
		Vector3(aabb.position.x + aabb.size.x, aabb.position.y, aabb.position.z + aabb.size.z),
		Vector3(aabb.position.x, aabb.position.y + aabb.size.y, aabb.position.z + aabb.size.z),
		Vector3(aabb.position.x + aabb.size.x, aabb.position.y + aabb.size.y, aabb.position.z + aabb.size.z),
	]:
		var world_corner := visual.to_global(corner)
		min_corner = min_corner.min(world_corner)
		max_corner = max_corner.max(world_corner)

	return AABB(min_corner, max_corner - min_corner)

func _ray_intersects_aabb(origin: Vector3, direction: Vector3, aabb: AABB) -> float:
	var t_min := -INF
	var t_max := INF
	var aabb_end := aabb.position + aabb.size

	for axis in 3:
		var dir_component := direction[axis]
		var origin_component := origin[axis]
		var min_component := aabb.position[axis]
		var max_component := aabb_end[axis]

		if abs(dir_component) < 0.000001:
			if origin_component < min_component or origin_component > max_component:
				return -1.0
			continue

		var t1 := (min_component - origin_component) / dir_component
		var t2 := (max_component - origin_component) / dir_component
		var axis_t_min = min(t1, t2)
		var axis_t_max = max(t1, t2)

		t_min = max(t_min, axis_t_min)
		t_max = min(t_max, axis_t_max)

		if t_min > t_max:
			return -1.0

	if t_max < 0.0:
		return -1.0

	return max(t_min, 0.0)

func _handle_human_cell_click(cell: Vector2i) -> void:
	match _interaction_stage:
		"placement":
			_try_submit_placement(cell)
		"select_worker":
			var worker_key := _worker_at_cell(cell)
			if _legal_worker_ids().has(worker_key):
				_selected_worker = worker_key
				_selected_move_target = INVALID_CELL
				_interaction_stage = "move_select"
				_refresh_human_highlights()
		"move_select":
			var clicked_worker := _worker_at_cell(cell)
			if _legal_worker_ids().has(clicked_worker):
				_reset_worker_preview()
				_selected_worker = clicked_worker
				_selected_move_target = INVALID_CELL
				_refresh_human_highlights()
				return
			if _legal_move_targets_for_worker(_selected_worker).has(cell):
				_selected_move_target = cell
				_start_move_preview(_selected_worker, cell)
		"build_select":
			var clicked_worker := _worker_at_cell(cell)
			if _legal_worker_ids().has(clicked_worker):
				_reset_worker_preview()
				_selected_worker = clicked_worker
				_selected_move_target = INVALID_CELL
				_interaction_stage = "move_select"
				_refresh_human_highlights()
				return
			_try_submit_turn(cell)

func _try_submit_placement(cell: Vector2i) -> void:
	for move in _legal_moves:
		if typeof(move) != TYPE_DICTIONARY or move.get("type", "") != "placement":
			continue
		if move.get("to", INVALID_CELL) != cell:
			continue
		_submit_human_move({
			"type": "placement",
			"worker": String(move.get("worker", "")),
			"to": [cell.x, cell.y],
		})
		return

func _try_submit_turn(build_cell: Vector2i) -> void:
	for move in _legal_moves:
		if typeof(move) != TYPE_DICTIONARY or move.get("type", "") != "turn":
			continue
		if String(move.get("worker", "")) != _selected_worker:
			continue
		if move.get("to", INVALID_CELL) != _selected_move_target:
			continue
		if move.get("build", INVALID_CELL) != build_cell:
			continue
		var from_coord: Vector2i = move.get("from", INVALID_CELL)
		var payload := {
			"type": "turn",
			"worker": _selected_worker,
			"from_coord": [from_coord.x, from_coord.y],
			"to": [_selected_move_target.x, _selected_move_target.y],
			"build": [build_cell.x, build_cell.y],
			"build_dome": bool(move.get("build_dome", false)),
		}
		_play_local_human_build_preview(build_cell)
		_submit_human_move_after_preview(payload)
		return

func _try_submit_winning_turn(worker_key: String, target_cell: Vector2i) -> void:
	for move in _legal_moves:
		if typeof(move) != TYPE_DICTIONARY or move.get("type", "") != "turn":
			continue
		if String(move.get("worker", "")) != worker_key:
			continue
		if move.get("to", INVALID_CELL) != target_cell:
			continue
		var from_coord: Vector2i = move.get("from", INVALID_CELL)
		var build_cell: Vector2i = move.get("build", INVALID_CELL)
		var payload := {
			"type": "turn",
			"worker": worker_key,
			"from_coord": [from_coord.x, from_coord.y],
			"to": [target_cell.x, target_cell.y],
			"build": [build_cell.x, build_cell.y],
			"build_dome": bool(move.get("build_dome", false)),
		}
		_submit_human_move(payload)
		return

func _submit_human_move(payload: Dictionary) -> void:
	_legal_moves = []
	_interaction_stage = ""
	Network.submit_human_move(payload)

func _submit_human_move_after_preview(payload: Dictionary) -> void:
	_pending_human_move_payload = payload.duplicate(true)
	_legal_moves = []
	_interaction_stage = "submitting_human_turn"
	_clear_cell_highlights()
	var timer := get_tree().create_timer(HUMAN_BUILD_SUBMIT_DELAY)
	timer.timeout.connect(func() -> void:
		if _pending_human_move_payload.is_empty():
			return
		var final_payload := _pending_human_move_payload
		_pending_human_move_payload = {}
		_submit_human_move(final_payload)
	)

func _play_local_human_build_preview(build_cell: Vector2i) -> void:
	var preview_board := _current_board.duplicate(true)
	if build_cell.x < 0 or build_cell.x >= preview_board.size():
		return
	if typeof(preview_board[build_cell.x]) != TYPE_ARRAY or build_cell.y < 0 or build_cell.y >= preview_board[build_cell.x].size():
		return
	preview_board[build_cell.x][build_cell.y] = int(preview_board[build_cell.x][build_cell.y]) + 1
	if int(preview_board[build_cell.x][build_cell.y]) >= 4:
		_dome_owners[build_cell] = _active_team
	_current_board = preview_board.duplicate(true)
	_apply_board(preview_board)
	_animate_local_build_preview(build_cell, _color_for_team(_active_team))

func _sync_dome_owners_with_board(board: Array) -> void:
	var valid_domes: Dictionary = {}
	for x in range(min(SIZE, board.size())):
		if typeof(board[x]) != TYPE_ARRAY:
			continue
		for y in range(min(SIZE, board[x].size())):
			if int(board[x][y]) >= 4:
				var cell := Vector2i(x, y)
				if _dome_owners.has(cell):
					valid_domes[cell] = _dome_owners[cell]
	_dome_owners = valid_domes

func _animate_local_build_preview(build_cell: Vector2i, team_color: Color) -> void:
	if build_cell.x < 0 or build_cell.x >= SIZE or build_cell.y < 0 or build_cell.y >= SIZE:
		return
	var stack: Node3D = stacks[build_cell.x][build_cell.y]
	if stack.get_child_count() == 0:
		return
	var built_node := stack.get_child(stack.get_child_count() - 1) as Node3D
	if built_node == null:
		return
	_animate_build_highlight(built_node, team_color)

func _board_height_at(cell: Vector2i) -> int:
	if cell.x < 0 or cell.x >= _current_board.size():
		return 0
	if typeof(_current_board[cell.x]) != TYPE_ARRAY or cell.y < 0 or cell.y >= _current_board[cell.x].size():
		return 0
	return int(_current_board[cell.x][cell.y])

func _start_move_preview(worker_key: String, target_cell: Vector2i) -> void:
	if not workers.has(worker_key):
		return
	var worker := workers[worker_key] as Node3D
	if worker == null or not is_instance_valid(worker):
		return

	_reset_worker_preview()

	_preview_worker_key = worker_key
	_preview_worker_origin_cell = worker_positions.get(worker_key, INVALID_CELL)
	_preview_worker_origin_position = worker.position
	_preview_move_target = target_cell
	_interaction_stage = "move_animating"
	_clear_cell_highlights()

	var target_position := _worker_target_position_for_cell(target_cell)
	_animate_worker_to(worker_key, worker, target_position, func() -> void:
		if _preview_worker_key != worker_key:
			return
		worker_positions[worker_key] = target_cell
		if _board_height_at(target_cell) >= 3:
			_interaction_stage = "submitting_human_turn"
			_clear_cell_highlights()
			_try_submit_winning_turn(worker_key, target_cell)
			return
		_interaction_stage = "build_select"
		_refresh_human_highlights()
	)

func _reset_worker_preview() -> void:
	if _preview_worker_key.is_empty():
		return

	if workers.has(_preview_worker_key):
		var worker := workers[_preview_worker_key] as Node3D
		if worker and is_instance_valid(worker):
			_kill_worker_tween(_preview_worker_key)
			worker.position = _preview_worker_origin_position

	if _preview_worker_origin_cell != INVALID_CELL:
		worker_positions[_preview_worker_key] = _preview_worker_origin_cell

	_preview_worker_key = ""
	_preview_worker_origin_cell = INVALID_CELL
	_preview_worker_origin_position = Vector3.ZERO
	_preview_move_target = INVALID_CELL

func _sync_preview_with_state(worker_state: Variant) -> void:
	if _preview_worker_key.is_empty():
		return
	if typeof(worker_state) != TYPE_DICTIONARY:
		_reset_worker_preview()
		return

	var state_workers := worker_state as Dictionary
	var preview_value = state_workers.get(_preview_worker_key)
	if preview_value is Vector2i and preview_value == _preview_move_target:
		_preview_worker_key = ""
		_preview_worker_origin_cell = INVALID_CELL
		_preview_worker_origin_position = Vector3.ZERO
		_preview_move_target = INVALID_CELL
	else:
		_reset_worker_preview()

func _worker_target_position_for_cell(cell: Vector2i) -> Vector3:
	var level_here := 0
	if cell.x >= 0 and cell.x < SIZE and cell.y >= 0 and cell.y < SIZE and cell.x < _current_board.size() and typeof(_current_board[cell.x]) == TYPE_ARRAY and cell.y < _current_board[cell.x].size():
		level_here = int(_current_board[cell.x][cell.y])
	var worker_offset_index = clamp(level_here, 0, WORKER_LEVEL_OFFSETS.size() - 1)
	var worker_y = LEVEL_HEIGHT * level_here + WORKER_LEVEL_OFFSETS[worker_offset_index]
	var cell_pos = cells[cell.x][cell.y].position
	return Vector3(cell_pos.x, worker_y, cell_pos.z)



func _on_reset_button_up() -> void:
	_clear_human_interaction()
	_winner_team = ""
	playing = false
	$UI/HBoxContainer/Play.text = "▶ Play"
	play_timer.stop()
	Network.reset_game({
		"one_type": AGENT_TYPE_VALUES[player_one_type_option.get_selected_id()],
		"two_type": AGENT_TYPE_VALUES[player_two_type_option.get_selected_id()],
	})

func _on_step_button_up() -> void:
	if not _winner_team.is_empty():
		return
	Network.step_game()
	
func _apply_block_colour(block: Node3D, height: int) -> void:
	var mesh := block.get_node_or_null("CSGBox3D")
	if mesh == null:
		return

	var mat := StandardMaterial3D.new()
	mat.albedo_color = LEVEL_COLORS[min(height, LEVEL_COLORS.size() - 1)]

	mesh.material_override = mat

var playing = false

func _on_play_button_up() -> void:
	if not _winner_team.is_empty():
		return
	if not playing:
		playing = true
		$UI/HBoxContainer/Play.text = "⏸ Pause"
		$UI/PlayTimer.start()
	else:
		$UI/HBoxContainer/Play.text = "▶ Play"
		playing = false
	
	
func _on_play_timer_timeout() -> void:
	if playing and _winner_team.is_empty():
		Network.step_game()
		play_timer.start()
	else:
		play_timer.stop()
		
	
