import argparse
from pathlib import Path

import torch

from flow_estimator import FlowEstimator


def _extract_state_dict(ckpt_obj):
	"""Return a state_dict regardless of common checkpoint wrappers."""
	if isinstance(ckpt_obj, dict):
		for key in ("state_dict", "model", "model_state_dict", "net", "weights"):
			if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
				return ckpt_obj[key]
	if isinstance(ckpt_obj, dict):
		return ckpt_obj
	raise TypeError("Unsupported checkpoint format. Expected dict-like checkpoint.")


def _normalize_key_prefix(key):
	# Handle checkpoints saved with DataParallel/DDP wrappers.
	if key.startswith("module."):
		return key[len("module."):]
	return key


def build_flow_estimator_state(src_state, dst_template):
	"""Map modelVFI keys to FlowEstimator keys and keep only matching tensors."""
	src_state = {_normalize_key_prefix(k): v for k, v in src_state.items()}
	remapped = {}
	copied = []

	for dst_key, dst_tensor in dst_template.items():
		candidate_keys = [
			dst_key,
			f"flowgen.{dst_key}" if not dst_key.startswith("flowgen.") else dst_key,
			f"up_sample.{dst_key}" if not dst_key.startswith("up_sample.") else dst_key,
		]

		src_tensor = None
		for ck in candidate_keys:
			if ck in src_state:
				src_tensor = src_state[ck]
				break

		if src_tensor is None:
			continue

		if src_tensor.shape != dst_tensor.shape:
			print(
				f"Skip shape mismatch: {dst_key} src={tuple(src_tensor.shape)} dst={tuple(dst_tensor.shape)}"
			)
			continue

		remapped[dst_key] = src_tensor
		copied.append(dst_key)

	return remapped, copied


def main():
	parser = argparse.ArgumentParser(
		description="Extract FlowEstimator weights from VTinker (modelVFI) checkpoint"
	)
	parser.add_argument(
		"--input",
		type=str,
		default="VTinker.pkl",
		help="Path to VTinker checkpoint (.pkl/.pth)",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="flow_estimator_from_vtinker.pth",
		help="Path to save extracted FlowEstimator weights",
	)
	args = parser.parse_args()

	input_path = Path(args.input)
	output_path = Path(args.output)

	if not input_path.exists():
		raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

	ckpt_obj = torch.load(str(input_path), map_location="cpu")
	src_state = _extract_state_dict(ckpt_obj)

	flow_estimator = FlowEstimator()
	dst_state = flow_estimator.state_dict()

	mapped_state, copied_keys = build_flow_estimator_state(src_state, dst_state)
	final_state = dict(dst_state)
	final_state.update(mapped_state)

	missing_keys = [k for k in dst_state.keys() if k not in mapped_state]

	load_info = flow_estimator.load_state_dict(final_state, strict=True)

	save_obj = {
		"state_dict": flow_estimator.state_dict(),
		"meta": {
			"source": str(input_path),
			"copied_params": len(copied_keys),
			"total_params": len(dst_state),
			"missing_from_source": missing_keys,
			"load_info": {
				"missing_keys": list(load_info.missing_keys),
				"unexpected_keys": list(load_info.unexpected_keys),
			},
		},
	}
	torch.save(save_obj, str(output_path))

	print(f"Saved FlowEstimator weights to: {output_path}")
	print(f"Copied params: {len(copied_keys)}/{len(dst_state)}")
	if missing_keys:
		print("Parameters not found in source checkpoint:")
		for k in missing_keys:
			print(f"  - {k}")
	else:
		print("All FlowEstimator parameters were copied from source checkpoint.")


if __name__ == "__main__":
	main()