from src.main import build_parser


def test_cli_help():
    parser = build_parser()
    assert parser.prog == "cvscope"


def test_image_command_exists():
    parser = build_parser()
    args = parser.parse_args(["image", "--input", "sample.png", "--task", "histogram"])
    assert args.command == "image"
    assert args.task == "histogram"
