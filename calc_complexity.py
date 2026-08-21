import time
import torch
from utils.get_model import create_model
from ptflops import get_model_complexity_info


#################### Config ####################
device = "cuda"

img_channel = 3
inp_shape = (3, 256, 256)

model_name = "NAFNet_EGD"
model_args = {"segmentation": True}

# --- for calc latency ---
warmup_runs = 20
num_runs = 100
################################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_model(model_name, model_args)

macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=False)

model = model.to(device)
model.eval()

dummy_input = torch.randn(1, *inp_shape).to(device)

with torch.no_grad():
    for _ in range(warmup_runs):
        _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_runs):
            _ = model(dummy_input)
        end_event.record()
        torch.cuda.synchronize()

        avg_latency_ms = start_event.elapsed_time(end_event) / num_runs

    else:
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model(dummy_input)
        end_time = time.perf_counter()

        avg_latency_ms = ((end_time - start_time) / num_runs) * 1000

print("\nModel: ", model_name)
print("Device:", device)
print("\nMacs: ", macs)
print("Params: ", params)
print(f"Latency: {avg_latency_ms:.2f} ms\n")