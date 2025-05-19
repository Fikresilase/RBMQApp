import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QFileDialog, QHBoxLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import os

# Median values for quantization
MEDIAN_VALUES = [
    4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92,
    100, 108, 116, 124, 132, 140, 148, 156, 164, 172,
    180, 188, 196, 204, 212, 220, 228, 236, 244, 252
]

# Create forward and reverse mappings
FORWARD_MAP = {}
REVERSE_MAP = {}

for idx, val in enumerate(MEDIAN_VALUES):
    FORWARD_MAP[val] = idx

for idx, val in enumerate(MEDIAN_VALUES):
    REVERSE_MAP[idx] = val


def find_closest_median(pixel_value):
    """Find closest median value for a given pixel."""
    idx = np.argmin(np.abs(np.array(MEDIAN_VALUES) - pixel_value))
    return MEDIAN_VALUES[idx]


def compress_image_array(image_array):
    """Quantize and compress image array to 5-bit per pixel."""
    height, width = image_array.shape
    compressed_bits = []

    for row in image_array:
        for pixel in row:
            # Find closest median
            quantized_pixel = find_closest_median(pixel)
            # Get index (0-31)
            index = FORWARD_MAP[quantized_pixel]
            # Convert to 5-bit binary string
            bin_str = format(index, '05b')
            compressed_bits.extend([int(bit) for bit in bin_str])

    # Pack bits into bytes
    packed_bytes = []
    for i in range(0, len(compressed_bits), 8):
        byte_bits = compressed_bits[i:i+8]
        # Pad with zeros if needed
        while len(byte_bits) < 8:
            byte_bits.append(0)
        byte_val = int(''.join(str(b) for b in byte_bits), 2)
        packed_bytes.append(byte_val)

    return bytes(packed_bytes), (height, width)


def decompress_to_image_array(compressed_data, shape):
    """Decompress data and rebuild image array."""
    height, width = shape
    bit_string = ''.join(f"{byte:08b}" for byte in compressed_data)
    
    # Extract 5-bit chunks
    pixels = []
    for i in range(0, len(bit_string), 5):
        chunk = bit_string[i:i+5]
        if len(chunk) < 5:
            chunk += '0' * (5 - len(chunk))  # pad if needed
        index = int(chunk, 2)
        pixel_value = REVERSE_MAP.get(index, 0)
        pixels.append(pixel_value)

    # Reshape into image
    image_array = np.array(pixels, dtype=np.uint8).reshape(height, width)
    return image_array


class ImageCompressorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Median Quantization Image Compressor")
        self.image_array = None
        self.compressed_data = None
        self.shape = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("Load a grayscale image or compressed file")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        btn_layout = QHBoxLayout()
        self.load_img_btn = QPushButton("Load Image")
        self.load_img_btn.clicked.connect(self.load_image)
        btn_layout.addWidget(self.load_img_btn)

        self.compress_btn = QPushButton("Compress & Save")
        self.compress_btn.clicked.connect(self.compress_and_save)
        self.compress_btn.setEnabled(False)
        btn_layout.addWidget(self.compress_btn)

        self.load_comp_btn = QPushButton("Load Compressed File")
        self.load_comp_btn.clicked.connect(self.load_compressed_file)
        btn_layout.addWidget(self.load_comp_btn)

        self.decompress_btn = QPushButton("Decompress Image")
        self.decompress_btn.clicked.connect(self.decompress_and_show)
        self.decompress_btn.setEnabled(False)
        btn_layout.addWidget(self.decompress_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if path:
            image = QImage(path)
            if image.format() != QImage.Format_Grayscale8:
                image = image.convertToFormat(QImage.Format_Grayscale8)

            self.image_array = np.zeros((image.height(), image.width()), dtype=np.uint8)
            for y in range(image.height()):
                for x in range(image.width()):
                    pixel = qGray(image.pixel(x, y))
                    self.image_array[y, x] = pixel

            self.label.setText("Image loaded. Ready to compress.")
            self.compress_btn.setEnabled(True)

    def compress_and_save(self):
        if self.image_array is None:
            return
        self.compressed_data, self.shape = compress_image_array(self.image_array)
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Compressed File", "", "Binary Files (*.bin)")
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(self.compressed_data)
            QMessageBox.information(self, "Success", "Image compressed and saved successfully.")

    def load_compressed_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Compressed File", "", "Binary Files (*.bin)")
        if path:
            with open(path, 'rb') as f:
                self.compressed_data = f.read()
            # Ask for dimensions
            dim, ok = QInputDialog.getText(self, "Image Dimensions", "Enter image dimensions (widthxheight):")
            if ok and 'x' in dim:
                try:
                    w, h = map(int, dim.split('x'))
                    self.shape = (h, w)
                    self.label.setText("Compressed file loaded. Ready to decompress.")
                    self.decompress_btn.setEnabled(True)
                except Exception as e:
                    print(e)
                    QMessageBox.warning(self, "Error", "Invalid dimensions entered.")

    def decompress_and_show(self):
        if self.compressed_data is None or self.shape is None:
            return
        image_array = decompress_to_image_array(self.compressed_data, self.shape)
        height, width = image_array.shape
        q_image = QImage(image_array.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image).scaled(400, 400, Qt.KeepAspectRatio)
        label = QLabel()
        label.setPixmap(pixmap)
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Reconstructed Image")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setFixedSize(500, 500)
        msg_box.layout().addWidget(label)
        msg_box.exec_()


def qGray(rgb):
    r = (rgb >> 16) & 0xFF
    g = (rgb >> 8) & 0xFF
    b = rgb & 0xFF
    return int(0.299 * r + 0.587 * g + 0.114 * b)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageCompressorApp()
    window.show()
    sys.exit(app.exec_())