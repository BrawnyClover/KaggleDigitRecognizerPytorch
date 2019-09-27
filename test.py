import so_compli
import numpy as np
import torch
import csv

net = torch.load('so_compli.pt')
f = open('submission.csv', 'w', newline='')
csv_writer = csv.writer(f)
test_raw_data = so_compli.load_test_data()

def get_image_tensor(idx):
    pixel_data = test_raw_data[idx][0:]
    pixel_data = np.expand_dims(pixel_data, 0)
    torch_pixel = torch.from_numpy(pixel_data)
    torch_float_pixel_data = torch_pixel.type(torch.FloatTensor)
    torch_float_pixel_data = torch_float_pixel_data / 255

    return torch_float_pixel_data

csv_writer.writerow( ["ImageId", "Label"])

for idx in range(28000):
    x = get_image_tensor(idx)
    value, indices = torch.max(net(x), 1)
    csv_writer.writerow( [idx+1, indices.data.numpy()[0]])