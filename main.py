import sys
import numpy as np
import heapq
import collections
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
                            QLabel, QRadioButton, QVBoxLayout, QWidget, QMessageBox,
                            QTableWidget, QTableWidgetItem, QDialog)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image

# Quantization median values
median_values = [4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124,
                 132, 140, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220, 228, 236, 244, 252]

# Huffman Tree Node
class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

# Helper functions for RLE + Huffman
def rle_encode(data):
    """Run-Length Encoding"""
    if len(data) == 0:
        return []
    
    encoded = []
    current_val = data[0]
    count = 1
    
    for val in data[1:]:
        if val == current_val:
            count += 1
        else:
            encoded.append((current_val, count))
            current_val = val
            count = 1
    encoded.append((current_val, count))
    return encoded

def rle_decode(encoded):
    """Run-Length Decoding"""
    decoded = []
    for val, count in encoded:
        decoded.extend([val] * count)
    return np.array(decoded, dtype=np.uint8)

def build_huffman_tree(freq_dict):
    """Build Huffman Tree from frequency dictionary"""
    heap = [HuffmanNode(char=char, freq=freq) for char, freq in freq_dict.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    return heap[0] if heap else None

def build_huffman_codes(root, current_code="", codes=None):
    """Build Huffman codes dictionary"""
    if codes is None:
        codes = {}
    
    if root is None:
        return
    
    if root.char is not None:
        codes[root.char] = current_code
        return
    
    build_huffman_codes(root.left, current_code + "0", codes)
    build_huffman_codes(root.right, current_code + "1", codes)
    
    return codes

def huffman_encode(data, codes):
    """Encode data using Huffman codes"""
    encoded_bits = ''.join([codes[item] for item in data])
    # Pad with zeros to make full bytes
    padding = 8 - len(encoded_bits) % 8
    if padding == 8:  # No padding needed if already multiple of 8
        padding = 0
    encoded_bits += '0' * padding
    
    # Convert to bytes
    byte_array = bytearray()
    for i in range(0, len(encoded_bits), 8):
        byte = encoded_bits[i:i+8]
        byte_array.append(int(byte, 2))
    
    return bytes(byte_array), padding

def huffman_decode(encoded_bytes, padding, root, original_length):
    """Decode Huffman encoded data"""
    bit_string = ''.join(f'{byte:08b}' for byte in encoded_bytes)
    bit_string = bit_string[:-padding] if padding else bit_string
    
    decoded = []
    current_node = root
    for bit in bit_string:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right
        
        if current_node.char is not None:
            decoded.append(current_node.char)
            current_node = root
            if len(decoded) == original_length:
                break
    
    return decoded

def save_compressed_data(data, filename):
    """Save compressed data with RLE + Huffman"""
    # Apply RLE
    rle_encoded = rle_encode(data.flatten())
    
    # Prepare data for Huffman
    flat_rle = []
    for val, count in rle_encoded:
        flat_rle.extend([val, count])
    
    # Build frequency dictionary
    freq = collections.Counter(flat_rle)
    
    # Build Huffman tree and codes
    root = build_huffman_tree(freq)
    if root is None:
        raise ValueError("Cannot build Huffman tree from empty data")
    
    codes = build_huffman_codes(root)
    if not codes:
        raise ValueError("No Huffman codes generated")
    
    # Huffman encode
    encoded_bytes, padding = huffman_encode(flat_rle, codes)
    
    # Save metadata and compressed data
    with open(filename, 'wb') as f:
        # Save original shape
        np.save(f, np.array(data.shape))
        # Save Huffman tree structure
        tree_structure = serialize_huffman_tree(root)
        np.save(f, np.array(len(tree_structure)))
        np.save(f, np.array(tree_structure))
        # Save padding
        np.save(f, np.array(padding))
        # Save original RLE length
        np.save(f, np.array(len(rle_encoded)))
        # Save compressed data
        f.write(encoded_bytes)

def load_compressed_data(filename):
    """Load and decompress data"""
    with open(filename, 'rb') as f:
        try:
            # Load original shape
            shape = tuple(np.load(f))
            # Load Huffman tree
            tree_size = np.load(f).item()
            tree_structure = np.load(f).tolist()
            if len(tree_structure) != tree_size:
                raise ValueError("Huffman tree structure size mismatch")
            root = deserialize_huffman_tree(iter(tree_structure))
            # Load padding
            padding = np.load(f).item()
            # Load original RLE length
            rle_length = np.load(f).item()
            # Load compressed data
            encoded_bytes = f.read()
        except Exception as e:
            raise ValueError(f"Corrupted file: {str(e)}")
    
    # Huffman decode
    decoded = huffman_decode(encoded_bytes, padding, root, rle_length * 2)
    if len(decoded) != rle_length * 2:
        raise ValueError("Decoded data length mismatch")
    
    # Reconstruct RLE
    rle_encoded = []
    for i in range(0, len(decoded), 2):
        if i+1 >= len(decoded):
            raise ValueError("Invalid RLE data")
        rle_encoded.append((decoded[i], decoded[i+1]))
    
    # RLE decode
    flat_data = rle_decode(rle_encoded)
    
    # Reshape to original
    try:
        return flat_data.reshape(shape)
    except ValueError as e:
        raise ValueError(f"Shape mismatch during reshape: {str(e)}")

def serialize_huffman_tree(root):
    """Serialize Huffman tree using pre-order traversal"""
    if root is None:
        return []
    
    if root.char is not None:
        return [1, root.char]
    
    return [0] + serialize_huffman_tree(root.left) + serialize_huffman_tree(root.right)

def deserialize_huffman_tree(data_iter):
    """Deserialize Huffman tree"""
    try:
        flag = next(data_iter)
    except StopIteration:
        return None
    
    if flag == 1:
        try:
            char = next(data_iter)
            return HuffmanNode(char=char)
        except StopIteration:
            raise ValueError("Incomplete Huffman tree data")
    
    left = deserialize_huffman_tree(data_iter)
    right = deserialize_huffman_tree(data_iter)
    return HuffmanNode(left=left, right=right)

# Image processing functions
def apply_median_quantization(img_array):
    """Apply median quantization to the image."""
    if img_array.dtype != np.uint8:
        raise ValueError("Input array must be of type uint8")
    
    quantized_array = np.zeros_like(img_array)
    for i, median in enumerate(median_values):
        lower_bound = i * 8
        upper_bound = lower_bound + 7
        quantized_array[(img_array >= lower_bound) & (img_array <= upper_bound)] = median
    return quantized_array

def apply_bit_reduction(img_array):
    """Reduce bit depth from 8 bits to 5 bits (32 levels)."""
    if img_array.dtype != np.uint8:
        raise ValueError("Input array must be of type uint8")
    return np.right_shift(img_array, 3)

def revert_bit_reduction(reduced_bit_image):
    """Revert using left shift (approximate reversion)."""
    if reduced_bit_image.dtype != np.uint8:
        raise ValueError("Input array must be of type uint8")
    return np.left_shift(reduced_bit_image, 3)

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_path = None
        self.current_image_array = None

    def initUI(self):
        self.setWindowTitle('Image Processing App')
        self.setGeometry(100, 100, 800, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout(self.central_widget)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; } 
            QLabel { font-size: 18px; font-weight: bold; color: #ffffff; } 
            QPushButton { 
                background-color: #4CAF50; color: white; border-radius: 10px; 
                padding: 12px; font-size: 16px; border: none; 
            } 
            QPushButton:hover { background-color: #45a049; } 
            QRadioButton { font-size: 16px; color: #ffffff; } 
            #imageLabel { 
                background-color: #333; border: 2px dashed #666; 
                min-width: 250px; min-height: 250px; 
            } 
            QDialog { background-color: #1e1e1e; }
            QTableWidget {
                background-color: #2d2d2d; color: #ffffff; gridline-color: #444;
            }
            QHeaderView::section {
                background-color: #333; color: white; padding: 4px; border: 1px solid #444;
            }
        """)
        
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(50, 50, 50, 50)
        
        self.label_instruction = QLabel('1. Select an image', self)
        self.label_instruction.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label_instruction)
        
        self.btn_choose = QPushButton('Choose Image', self)
        self.btn_choose.clicked.connect(self.choose_image)
        self.layout.addWidget(self.btn_choose, alignment=Qt.AlignCenter)
        
        self.label_image = QLabel(self)
        self.label_image.setObjectName("imageLabel")
        self.label_image.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label_image, alignment=Qt.AlignCenter)
        
        self.label_options = QLabel('2. Choose an option', self)
        self.label_options.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label_options)
        
        self.option1 = QRadioButton('Apply Median Quantization', self)
        self.option2 = QRadioButton('Apply Quantization + Bit Reduction', self)
        self.option3 = QRadioButton('Revert to Quantized Image', self)
        self.layout.addWidget(self.option1, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.option2, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.option3, alignment=Qt.AlignCenter)
        
        self.btn_process = QPushButton('Process and Save Image', self)
        self.btn_process.clicked.connect(self.process_and_save_image)
        self.layout.addWidget(self.btn_process, alignment=Qt.AlignCenter)

    def choose_image(self):
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(
            self, 'Select Image', '', 
            'Image Files (*.png *.jpg *.jpeg);;All Files (*)', options=options)
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.label_image.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
            try:
                img = Image.open(self.image_path)
                self.current_image_array = np.array(img)
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load image: {str(e)}')
                self.image_path = None
                self.current_image_array = None

    def show_values_dialog(self, quantized_values, reduced_values):
        dialog = ValueDisplayDialog(quantized_values, reduced_values, self)
        dialog.exec_()

    def process_and_save_image(self):
        if not self.image_path:
            QMessageBox.warning(self, 'Error', 'Please select an image first.')
            return
        
        option = 1 if self.option1.isChecked() else 2 if self.option2.isChecked() else 3 if self.option3.isChecked() else None
        if option is None:
            QMessageBox.warning(self, 'Error', 'Please choose an option.')
            return
        
        try:
            if self.current_image_array is None:
                img = Image.open(self.image_path)
                img_array = np.array(img)
            else:
                img_array = self.current_image_array
            
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.ndim == 3 and img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]  # Remove alpha channel if present
            
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            
            if option == 1:
                # Median quantization only
                r_quant = apply_median_quantization(r)
                g_quant = apply_median_quantization(g)
                b_quant = apply_median_quantization(b)
                processed_img_array = np.stack([r_quant, g_quant, b_quant], axis=-1)
                
                save_as, _ = QFileDialog.getSaveFileName(
                    self, 'Save Quantized Image', '', 
                    'PNG files (*.png);;JPEG files (*.jpg *.jpeg);;All Files (*)')
                if save_as:
                    Image.fromarray(processed_img_array.astype(np.uint8)).save(save_as)
                    QMessageBox.information(self, 'Success', f'Quantized image saved to {save_as}')
            
            elif option == 2:
                # Quantization + bit reduction + compression
                r_quant = apply_median_quantization(r)
                g_quant = apply_median_quantization(g)
                b_quant = apply_median_quantization(b)
                r_reduced = apply_bit_reduction(r_quant)
                g_reduced = apply_bit_reduction(g_quant)
                b_reduced = apply_bit_reduction(b_quant)
                processed_img_array = np.stack([r_reduced, g_reduced, b_reduced], axis=-1)
                
                # Show quantization values
                reduced_values = [apply_bit_reduction(np.array([val]))[0] for val in median_values]
                self.show_values_dialog(median_values, reduced_values)
                
                # Save compressed
                save_as, _ = QFileDialog.getSaveFileName(
                    self, 'Save Compressed Data', '', 
                    'Binary files (*.bin)')
                
                if save_as:
                    if not save_as.lower().endswith('.bin'):
                        save_as += '.bin'
                    try:
                        save_compressed_data(processed_img_array, save_as)
                        QMessageBox.information(self, 'Success', f'Compressed data saved to {save_as}')
                    except Exception as e:
                        QMessageBox.critical(self, 'Error', f"Failed to save compressed data: {str(e)}")
            
            elif option == 3:
                # Revert compressed data
                file_path, _ = QFileDialog.getOpenFileName(
                    self, 'Open Compressed Data', '', 
                    'Binary files (*.bin)')
                
                if file_path:
                    try:
                        loaded = load_compressed_data(file_path)
                        
                        # Verify dimensions
                        if loaded.ndim != 3 or loaded.shape[2] != 3:
                            raise ValueError("Invalid file format - expected 3-channel image data")
                        
                        r_reverted = revert_bit_reduction(loaded[:,:,0])
                        g_reverted = revert_bit_reduction(loaded[:,:,1])
                        b_reverted = revert_bit_reduction(loaded[:,:,2])
                        processed_img_array = np.stack([r_reverted, g_reverted, b_reverted], axis=-1)
                        
                        # Save the reverted image
                        save_as, _ = QFileDialog.getSaveFileName(
                            self, 'Save Reverted Image', '', 
                            'PNG files (*.png);;JPEG files (*.jpg *.jpeg);;All Files (*)')
                        if save_as:
                            Image.fromarray(processed_img_array.astype(np.uint8)).save(save_as)
                            QMessageBox.information(self, 'Success', f'Reverted image saved to {save_as}')
                    
                    except Exception as e:
                        QMessageBox.critical(self, 'Error', f"Failed to process binary file: {str(e)}")
        
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"An error occurred: {str(e)}")

class ValueDisplayDialog(QDialog):
    def __init__(self, quantized_values, reduced_values, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quantization and Bit Reduction Values")
        self.setGeometry(200, 200, 600, 400)
        
        layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Original Range", "Quantized Value", "Reduced Value"])
        self.table.setRowCount(len(quantized_values))
        
        for i in range(len(quantized_values)):
            lower_bound = i * 8
            upper_bound = lower_bound + 7
            range_item = QTableWidgetItem(f"{lower_bound}-{upper_bound}")
            quant_item = QTableWidgetItem(str(quantized_values[i]))
            reduced_item = QTableWidgetItem(str(reduced_values[i]))
            
            self.table.setItem(i, 0, range_item)
            self.table.setItem(i, 1, quant_item)
            self.table.setItem(i, 2, reduced_item)
        
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    ex.show()
    sys.exit(app.exec_())