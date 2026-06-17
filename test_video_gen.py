import imageio
import numpy as np

# Create a simple dummy red frame
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
dummy_frame[:] = [0, 0, 255] # Pure Red

# Try to save it
print("Attempting to save test video...")
try:
    writer = imageio.get_writer('test_output.mp4', fps=10, codec='libx264')
    writer.append_data(dummy_frame)
    writer.close()
    print("SUCCESS: Video created!")
except Exception as e:
    print(f"FAILED: {e}")