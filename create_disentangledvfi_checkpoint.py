import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import torch


def extract_state_dict(payload):
    if isinstance(payload, (dict, OrderedDict)):
        if all(torch.is_tensor(v) for v in payload.values()):
            return payload

        for key in ["state_dict", "model_state_dict", "model", "net", "params"]:
            if key in payload and isinstance(payload[key], (dict, OrderedDict)):
                candidate = payload[key]
                if all(torch.is_tensor(v) for v in candidate.values()):
                    return candidate

    raise ValueError("Could not find a valid state_dict in checkpoint payload.")


def strip_module_prefix(key: str) -> str:
    if key.startswith("module."):
        return key[len("module.") :]
    return key


def remap_flow_key(key: str) -> str:
    key = strip_module_prefix(key)
    if key.startswith("flow_estimator."):
        return key
    if key.startswith("flowgen.") or key.startswith("up_sample."):
        return "flow_estimator." + key
    if key.startswith("flow_base") or key.startswith("flow_large"):
        return ""
    return ""


def remap_vtinker_key(key: str) -> str:
    key = strip_module_prefix(key)

    replacements = [
        ("main_model.Ehead_warp.", "synthesis_net.ehead_warp."),
        ("main_model.lite_net.", "synthesis_net.lite_unet."),
        ("main_model.mid_net.", "synthesis_net.unet."),
    ]

    for src, dst in replacements:
        if key.startswith(src):
            return dst + key[len(src) :]

    return ""


def remap_state_dict(
    state_dict: Dict[str, torch.Tensor],
    mapper,
) -> Tuple[Dict[str, torch.Tensor], int]:
    mapped = {}
    skipped = 0
    for key, val in state_dict.items():
        target = mapper(key)
        if target:
            mapped[target] = val
        else:
            skipped += 1
    return mapped, skipped


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create an initial DisentangledVFI checkpoint from flow_estimator.pth "
            "and VTinker.pkl weights."
        )
    )
    parser.add_argument(
        "--flow-ckpt",
        default="weights/flow_estimator.pth",
        help="Path to flow estimator checkpoint.",
    )
    parser.add_argument(
        "--vtinker-ckpt",
        default="bin/VTinker.pkl",
        help="Path to VTinker checkpoint.",
    )
    parser.add_argument(
        "--out",
        default="weights/DisentangledVFI_init.pth",
        help="Output checkpoint path.",
    )
    args = parser.parse_args()

    flow_path = Path(args.flow_ckpt)
    vtinker_path = Path(args.vtinker_ckpt)
    out_path = Path(args.out)

    if not flow_path.exists():
        raise FileNotFoundError(f"Flow checkpoint not found: {flow_path}")
    if not vtinker_path.exists():
        raise FileNotFoundError(f"VTinker checkpoint not found: {vtinker_path}")

    flow_payload = torch.load(flow_path, map_location="cpu")
    vtinker_payload = torch.load(vtinker_path, map_location="cpu")

    flow_sd = extract_state_dict(flow_payload)
    vtinker_sd = extract_state_dict(vtinker_payload)

    mapped_flow, skipped_flow = remap_state_dict(flow_sd, remap_flow_key)
    mapped_vtinker, skipped_vtinker = remap_state_dict(vtinker_sd, remap_vtinker_key)

    merged_state = {}
    merged_state.update(mapped_flow)
    merged_state.update(mapped_vtinker)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dict = {
        "state_dict": merged_state,
        "meta": {
            "source_flow_ckpt": str(flow_path),
            "source_vtinker_ckpt": str(vtinker_path),
            "num_params_flow": len(mapped_flow),
            "num_params_vtinker": len(mapped_vtinker),
            "skipped_keys_flow": skipped_flow,
            "skipped_keys_vtinker": skipped_vtinker,
            "note": (
                "This is a partial initialization checkpoint. "
                "Load with strict=False into DisentangledVFI."
            ),
        },
    }
    print(dict["meta"])

    # counter number of parameters in the merged state dict
    total_params = sum(v.numel() for v in merged_state.values())
    print(f"Total parameters in merged state dict: {total_params}")
    torch.save(
        {
            "state_dict": merged_state
        },
        out_path,
    )

    print("Saved:", out_path)
    print("Mapped flow keys:", len(mapped_flow), "Skipped:", skipped_flow)
    print("Mapped VTinker keys:", len(mapped_vtinker), "Skipped:", skipped_vtinker)


if __name__ == "__main__":
    main()
