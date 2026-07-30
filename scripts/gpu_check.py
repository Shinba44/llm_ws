"""GPUが使えるか確認する。

    pixi run -e gpu gpu-check

古いGPU（Pascal 等）では、torch のビルドに実行可能なカーネルが含まれて
いるかが決定的に重要なので、それを検査する。

判定の根拠は2つ:
  1. arch_list との突合（CUDAの前方互換規則を考慮する。下記 compatible_archs）
  2. 実際に行列積を1回走らせる ← こちらが最終的な証拠
"""

import torch

# compute capability から世代名を引く
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


def parse_arch(tag: str) -> tuple[int, int] | None:
    """'sm_86' -> (8, 6) / 'sm_120' -> (12, 0)。末尾1桁がminor。"""
    if not tag.startswith("sm_"):
        return None
    digits = tag[3:].rstrip("af+")  # sm_90a のような接尾辞を落とす
    if not digits.isdigit() or len(digits) < 2:
        return None
    return int(digits[:-1]), int(digits[-1])


def compatible_archs(arch_list: list[str], major: int, minor: int) -> list[str]:
    """このデバイスで実行できる cubin を返す。

    CUDAの binary compatibility:
      compute capability X.y 向けの cubin は X.z (z >= y) のデバイスで動く。
    したがってメジャーが一致し、かつ cubin 側のマイナーがデバイス以下なら実行可能。
    例: sm_60 のカーネルは cc 6.1 の GTX 1080 Ti で動作する。
    """
    out = []
    for tag in arch_list:
        parsed = parse_arch(tag)
        if parsed and parsed[0] == major and parsed[1] <= minor:
            out.append(tag)
    return out


def native_bf16() -> bool:
    """エミュレーションを除いた、ハードウェアとしてのbf16対応。

    torch.cuda.is_bf16_supported() は既定でエミュレーションを含めて判定するため、
    Pascal でも True を返してしまう。
    """
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # 古いtorchには引数が無い
        return torch.cuda.is_bf16_supported()


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

    arch_list = torch.cuda.get_arch_list()
    print("torchが同梱するカーネル:", " ".join(arch_list))
    print()

    count = torch.cuda.device_count()
    print(f"GPU {count}枚")

    warnings: list[str] = []
    bf16 = native_bf16()

    for i in range(count):
        p = torch.cuda.get_device_properties(i)
        cc = (p.major, p.minor)
        gen = GENERATION.get(cc, "不明")
        free = torch.cuda.mem_get_info(i)[0] / 1024**3
        usable = compatible_archs(arch_list, p.major, p.minor)

        print(f"\n[{i}] {p.name}")
        print(f"    世代            : {gen} (compute capability {p.major}.{p.minor})")
        print(f"    VRAM            : {p.total_memory / 1024**3:.1f} GB（空き {free:.1f} GB）")
        print(f"    bf16(ネイティブ): {bf16}")
        print(f"    目安スコープ    : {scope_for(free)}  (docs/00_PLAN.md §7-2)")

        if usable:
            print(f"    実行可能カーネル: ✅ {' '.join(usable)}")
        else:
            print("    実行可能カーネル: ❌ 見つからない")
            warnings.append(
                f"GPU{i}: sm_{p.major}{p.minor} で実行できるカーネルがtorchに無い。"
                "より古いCUDA indexを試す（docs/01_SETUP.md §4.3）"
            )

        # 最終的な証拠。ここが通れば実際に計算できる
        try:
            x = torch.randn(512, 512, device=f"cuda:{i}")
            _ = (x @ x).sum().item()
            print("    行列積テスト    : ✅ 成功")
        except Exception as e:  # noqa: BLE001
            print("    行列積テスト    : ❌ 失敗")
            print("       ", type(e).__name__, str(e)[:160])
            warnings.append(f"GPU{i}: 実際の計算に失敗した。torchのCUDAビルドが非対応")

        if cc < (7, 0):
            warnings.append(
                f"GPU{i}: {gen}世代。Tensor Coreが無くbf16もネイティブ非対応。"
                "学習はfp32主体で組む。FlashAttentionは利用不可。"
                "QLoRA(bitsandbytes 4bit)は要検証"
            )

    if warnings:
        print("\n--- 注意 ---")
        for w in warnings:
            print(" *", w)


if __name__ == "__main__":
    main()
