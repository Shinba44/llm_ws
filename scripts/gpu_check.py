"""GPUが使えるか確認する。

    pixi run -e gpu gpu-check

`cuda available: False` になる場合は docs/01_SETUP.md §9 を参照。
"""

import torch


def main() -> None:
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("→ docs/01_SETUP.md §9 のトラブルシューティングを確認してください")
        return

    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1024**3
    print("device:", props.name)
    print(f"VRAM(GB): {vram:.1f}")
    print("bf16 supported:", torch.cuda.is_bf16_supported())

    # docs/00_PLAN.md §7 のスコープ表に対応させる
    if vram < 6.5:
        scope = "事前学習 〜30Mパラメータ / FT 0.5B QLoRA"
    elif vram < 13:
        scope = "事前学習 〜150Mパラメータ / FT 1〜3B QLoRA"
    else:
        scope = "事前学習 〜500Mパラメータ / FT 7〜8B QLoRA"
    print("目安スコープ:", scope, "(docs/00_PLAN.md §7)")


if __name__ == "__main__":
    main()
