extends Node3D

@export var target_path: NodePath
@export var distance := 25.0
@export var min_distance := 40.0
@export var max_distance := 60.0

@export var pitch_deg := -25.0
@export var min_pitch := -50.0
@export var max_pitch := -10.0

@export var rotate_sensitivity := 0.25
@export var zoom_sensitivity := 2.0

@onready var pitch_node: Node3D = $Pitch
@onready var cam: Camera3D = $Pitch/Camera3D
@onready var target: Node3D = get_node(target_path) as Node3D

var _dragging := false

func _ready() -> void:
	_update_rig()

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			_dragging = event.pressed
		elif event.button_index == MOUSE_BUTTON_WHEEL_UP and event.pressed:
			distance = clamp(distance - zoom_sensitivity, min_distance, max_distance)
			_update_rig()
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN and event.pressed:
			distance = clamp(distance + zoom_sensitivity, min_distance, max_distance)
			_update_rig()

	if event is InputEventMouseMotion and _dragging:
		rotation.y -= deg_to_rad(event.relative.x * rotate_sensitivity)
		pitch_deg = clamp(pitch_deg - event.relative.y * rotate_sensitivity, min_pitch, max_pitch)
		_update_rig()

func _update_rig() -> void:
	if target:
		global_position = target.global_position 

	pitch_node.rotation.x = deg_to_rad(pitch_deg)
	cam.position = Vector3(0, 0, distance)
	cam.look_at(global_position, Vector3.UP)
