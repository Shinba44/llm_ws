"""GPUが使えるか確認する。

    pixi run -e gpu gpu-check

`cuda available: False` になる場合は docs/01_SETUP.md §9 を参照。
古いGPU（Pascal 等）では torch のビルドに該当アーキテクチャの
カーネルが含まれているかが決定的に重要なので、それも検査する。
"""

import torch

# アーキテクチャ世代の判定に使う（compute capability の major, minor）
GENERATION = {
    (6, 0): "Pascal",
    (6, 1): "Pascal",
    (7, 0): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 9): "Ada Lovelace",
    (9, 0): "Hopper",
    (10, 0): "Blackwell",
    (12, 0): "Blackwell",
}


def scope_for(vram_gb: float) -> str:
    """docs/00_PLAN.md §7-2 のスコープ表に対応させる。"""
    if vram_gb < 6.5:
        return "事前学習 〜30Mパラメータ / FT 0.5〜1B"
    if vram_gb < 13:
        return "事前学習 〜150Mパラメータ / FT 1〜3B"
    return "事前学習 〜500Mパラメータ / FT 7〜8B"


def main() -> None:
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("→ docs/01_SETUP.md §9 のトラブルシューティングを確認してください")
        return

    # このtorchビルドに含まれるカーネルのアーキテクチャ一覧
    arch_list = torch.cuda.get_arch_list()
    print("torchが対応するarch:", " ".join(arch_list))
    print()

    count = torch.cuda.device_count()
    print(f"GPU {count}枚")

    warnings: list[str] = []

    for i in range(count):
        p = torch.cuda.get_device_properties(i)
        cc = (p.major, p.minor)
        gen = GENERATION.get(cc, "不明")
        vram = p.total_memory / 1024**3
        free = torch.cuda.mem_get_info(i)[0] / 1024**3

        print(f"\n[{i}] {p.name}")
        print(f"    世代            : {gen} (compute capability {p.major}.{p.minor})")
        print(f"    VRAM            : {vram:.1f} GB（空き {free:.1f} GB）")
        print(f"    bf16            : {torch.cuda.is_bf16_supported()}")
        print(f"    目安スコープ    : {scope_for(free)}  (docs/00_PLAN.md §7-2)")

        # ★決定的な検査: このGPU向けのカーネルがtorchに入っているか
        tag = f"sm_{p.major}{p.minor}"
        if tag in arch_list:
            print(f"    カーネル({tag}) : ✅ 同梱されている")
        else:
            print(f"    カーネル({tag}) : ❌ 含まれていない")
            warnings.append(
                f"GPU{i}: このtorchビルドに {tag} のカーネルが無い。"
                "実行時にエラーになる。より古いCUDA indexを試す（01_SETUP.md §4.3）"
            )

        if cc < (7, 5):
            warnings.append(
                f"GPU{i}: {gen}世代。Tensor Coreが無くbf16も使えない。"
                "学習はfp32主体、FlashAttentionは利用不可。QLoRA(bitsandbytes 4bit)も要検証"
            )

    # 実際に計算を1回走らせる。ここで落ちるならカーネル非対応が確定
    print()
    try:
        x = torch.randn(256, 256, device="cuda")
        _ = (x @ x).sum().item()
        print("行列積テスト: ✅ 成功")
    except Exception as e:  # noqa: BLE001
        print("行列積テスト: ❌ 失敗")
        print("   ", type(e).__name__, str(e)[:200])
        warnings.append("実際の計算に失敗した。torchのCUDAビルドがこのGPUに対応していない")

    if warnings:
        print("\n--- 注意 ---")
        for w in warnings:
            print(" *", w)


if __name__ == "__main__":
    main()
