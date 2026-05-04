extends TextureRect

@export var subviewport_path: NodePath
@onready var sv: SubViewport = get_node(subviewport_path) as SubViewport

func _ready() -> void:
	gui_input.connect(_on_gui_input)

func _on_gui_input(event: InputEvent) -> void:
	if sv == null:
		return

	var sv_size := Vector2(sv.size)

	if event is InputEventMouse:
		var e := event.duplicate()

		var local_pos := get_local_mouse_position()

		if local_pos.x < 0.0 or local_pos.y < 0.0 or local_pos.x > size.x or local_pos.y > size.y:
			return

		var uv := local_pos / size

		e.position = uv * sv_size

		if e is InputEventMouseMotion:
			var mm := e as InputEventMouseMotion
			mm.relative = mm.relative * (sv_size / size)

		sv.push_input(e)
	else:
		sv.push_input(event)
