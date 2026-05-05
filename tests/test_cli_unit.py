from evoresearcher.main import build_parser


def test_global_model_flag_is_parsed():
    args = build_parser().parse_args(
        [
            "--goal",
            "test",
            "--global-model",
            "deepseek-v4-flash",
        ]
    )

    assert args.global_model == "deepseek-v4-flash"
