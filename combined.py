import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, 
                            QLabel, QVBoxLayout, QWidget, QHBoxLayout, QStackedWidget,
                            QMessageBox, QRadioButton, QGroupBox)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt
from PIL import Image
from scipy.io import savemat, loadmat
import zlib

# Define the quantization median values for each group (32 groups with a width of 8)
median_values = [4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124,
                 132, 140, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220, 228, 236, 244, 252]

def print_pixel_samples(title, array):
    """Print sample 5x5 block of pixel values"""
    print(f"\n{title}:")
    print(array[:5, :5])

def apply_median_quantization(img_array):
    """Apply median quantization to the image.""" 
    quantized_array = np.zeros_like(img_array)
    for i, median in enumerate(median_values):
        lower_bound = i * 8
        upper_bound = lower_bound + 7
        quantized_array[(img_array >= lower_bound) & (img_array <= upper_bound)] = median
    return quantized_array

def apply_bit_reduction(img_array):
    """Reduce bit depth from 8 bits to 5 bits (32 levels)."""
    reduced_bit_image = np.right_shift(img_array, 3)  # Bit reduction
    return reduced_bit_image

def revert_bit_reduction(reduced_bit_image):
    """Revert from bit reduction (5 bits) to original quantized image."""
    reverted_image = np.left_shift(reduced_bit_image, 3)+4  # Reverting bit depth
    return reverted_image

def pack_5bit_array(data):
    """Pack 5-bit values into bytes (8 values -> 5 bytes)"""
    flat = data.flatten()
    extra = len(flat) % 8
    if extra:
        flat = np.pad(flat, (0, 8 - extra), 'constant')
    
    packed = np.zeros((len(flat) * 5 // 8,), dtype=np.uint8)
    
    for i in range(len(flat) // 8):
        vals = flat[i*8:(i+1)*8]
        packed[i*5] = (vals[0] << 3) | (vals[1] >> 2)
        packed[i*5+1] = ((vals[1] & 0x03) << 6) | (vals[2] << 1) | (vals[3] >> 4)
        packed[i*5+2] = ((vals[3] & 0x0F) << 4) | (vals[4] >> 1)
        packed[i*5+3] = ((vals[4] & 0x01) << 7) | (vals[5] << 2) | (vals[6] >> 3)
        packed[i*5+4] = ((vals[6] & 0x07) << 5) | vals[7]
    
    return packed, data.shape

def unpack_5bit_bytes(packed_data, original_shape):
    """Unpack bit-packed data back to 5-bit values"""
    total_values = original_shape[0] * original_shape[1] * original_shape[2]
    unpacked = np.zeros(total_values, dtype=np.uint8)
    
    # Calculate needed bytes (5 bytes per 8 values)
    needed_bytes = (total_values * 5 + 7) // 8
    packed_data = packed_data[:needed_bytes]
    
    for i in range(len(packed_data) // 5):
        byte0 = packed_data[i*5]
        byte1 = packed_data[i*5+1]
        byte2 = packed_data[i*5+2]
        byte3 = packed_data[i*5+3]
        byte4 = packed_data[i*5+4]
        
        # Unpack 8 values from 5 bytes
        unpacked[i*8] = byte0 >> 3
        unpacked[i*8+1] = ((byte0 & 0x07) << 2) | (byte1 >> 6)
        unpacked[i*8+2] = (byte1 >> 1) & 0x1F
        unpacked[i*8+3] = ((byte1 & 0x01) << 4) | (byte2 >> 4)
        unpacked[i*8+4] = ((byte2 & 0x0F) << 1) | (byte3 >> 7)
        unpacked[i*8+5] = (byte3 >> 2) & 0x1F
        unpacked[i*8+6] = ((byte3 & 0x03) << 3) | (byte4 >> 5)
        unpacked[i*8+7] = byte4 & 0x1F
    
    return unpacked[:total_values].reshape(original_shape)

class DayToDayCompressPage(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        self.label_instruction = QLabel('1. Select an image')
        self.label_instruction.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_instruction)
        
        self.btn_choose = QPushButton('Choose Image')
        self.btn_choose.clicked.connect(self.choose_image)
        layout.addWidget(self.btn_choose, alignment=Qt.AlignCenter)
        
        self.label_image = QLabel()
        self.label_image.setObjectName("imageLabel")
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setMinimumSize(400, 300)
        layout.addWidget(self.label_image, alignment=Qt.AlignCenter)
        
        self.label_options = QLabel('2. Choose an option')
        self.label_options.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_options)
        
        self.option1 = QRadioButton('Apply Median Quantization')
        self.option2 = QRadioButton('Apply Quantization + Bit Reduction')
        layout.addWidget(self.option1, alignment=Qt.AlignCenter)
        layout.addWidget(self.option2, alignment=Qt.AlignCenter)
        
        self.btn_process = QPushButton('Process and Save Image')
        self.btn_process.clicked.connect(self.process_and_save_image)
        layout.addWidget(self.btn_process, alignment=Qt.AlignCenter)
        
        self.setLayout(layout)
    
    def choose_image(self):
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(
            self, 'Select Image', '', 
            'Image Files (*.png *.jpg *.jpeg);;All Files (*)', 
            options=options)
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.label_image.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def process_and_save_image(self):
        if not self.image_path:
            QMessageBox.warning(self, 'Error', 'Please select an image first.')
            return
        
        option = 1 if self.option1.isChecked() else 2 if self.option2.isChecked() else None
        if option is None:
            QMessageBox.warning(self, 'Error', 'Please choose an option.')
            return
        
        try:
            img = Image.open(self.image_path)
            img_array = np.array(img)  # Get RGB values
            
            if img_array.ndim == 2:  # If grayscale, convert to RGB
                img_array = np.stack([img_array] * 3, axis=-1)
            
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            
            if option == 1:
                r_quant = apply_median_quantization(r)
                g_quant = apply_median_quantization(g)
                b_quant = apply_median_quantization(b)
                processed_img_array = np.stack([r_quant, g_quant, b_quant], axis=-1)
            elif option == 2:
                r_quant = apply_median_quantization(r)
                g_quant = apply_median_quantization(g)
                b_quant = apply_median_quantization(b)
                r_reduced = apply_bit_reduction(r_quant)
                g_reduced = apply_bit_reduction(g_quant)
                b_reduced = apply_bit_reduction(b_quant)
                processed_img_array = np.stack([r_reduced, g_reduced, b_reduced], axis=-1)
            
            processed_img = Image.fromarray(processed_img_array.astype(np.uint8))
            
            save_as, _ = QFileDialog.getSaveFileName(
                self, 'Save Image', '', 
                'PNG files (*.png);;JPEG files (*.jpeg);;All Files (*)')
            if save_as:
                processed_img.save(save_as)
                QMessageBox.information(self, 'Success', f'Image saved as {save_as}')
        
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"An error occurred: {e}")

class DayToDayDecompressPage(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        layout.addWidget(self.image_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_load = QPushButton('Load Image')
        self.btn_load.clicked.connect(self.load_image)
        
        self.btn_process = QPushButton('Process Bits')
        self.btn_process.clicked.connect(self.process_bits)
        self.btn_process.setEnabled(False)
        
        self.btn_save = QPushButton('Save Processed Image')
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False)
        
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_process)
        btn_layout.addWidget(self.btn_save)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select Image', '', 
            'Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)', 
            options=options)
            
        if file_path:
            self.image_path = file_path
            try:
                # Load with PIL first to handle all formats
                pil_img = Image.open(file_path)
                self.original_image = np.array(pil_img)
                
                # Convert to QPixmap for display
                if pil_img.mode == 'RGB':
                    bytes_per_line = 3 * pil_img.width
                    qimage = QImage(pil_img.tobytes(), pil_img.width, pil_img.height, 
                                   bytes_per_line, QImage.Format_RGB888)
                elif pil_img.mode == 'RGBA':
                    bytes_per_line = 4 * pil_img.width
                    qimage = QImage(pil_img.tobytes(), pil_img.width, pil_img.height, 
                                   bytes_per_line, QImage.Format_RGBA8888)
                else:  # Grayscale
                    bytes_per_line = pil_img.width
                    qimage = QImage(pil_img.tobytes(), pil_img.width, pil_img.height, 
                                   bytes_per_line, QImage.Format_Grayscale8)
                
                pixmap = QPixmap.fromImage(qimage)
                self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.btn_process.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, 'Error', f"Error loading image: {e}")
    
    def process_bits(self):
        if self.original_image is None:
            return
            
        try:
            # Process bits: shift left 3 and add 4
            self.processed_image = (self.original_image << 3) + 4
            
            # Clip values to 0-255 range
            self.processed_image = np.clip(self.processed_image, 0, 255).astype(np.uint8)
            
            # Display the processed image
            if len(self.processed_image.shape) == 2:  # Grayscale
                bytes_per_line = self.processed_image.shape[1]
                qimage = QImage(self.processed_image.data, self.processed_image.shape[1], 
                              self.processed_image.shape[0], bytes_per_line, QImage.Format_Grayscale8)
            else:  # Color
                bytes_per_line = 3 * self.processed_image.shape[1]
                qimage = QImage(self.processed_image.data, self.processed_image.shape[1], 
                              self.processed_image.shape[0], bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.btn_save.setEnabled(True)
            QMessageBox.information(self, 'Success', 'Processing complete!')
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"Error processing image: {e}")
    
    def save_image(self):
        if self.processed_image is None:
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Image', '', 
            'PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)', 
            options=options)
            
        if file_path:
            try:
                # Convert numpy array to PIL Image
                if len(self.processed_image.shape) == 2:  # Grayscale
                    processed_img = Image.fromarray(self.processed_image, mode='L')
                else:  # Color
                    processed_img = Image.fromarray(self.processed_image, mode='RGB')
                
                # Save with appropriate format based on extension
                processed_img.save(file_path)
                QMessageBox.information(self, 'Success', f'Image saved to: {file_path}')
                
            except Exception as e:
                QMessageBox.critical(self, 'Error', f"Error saving image: {e}")

class ScientificCompressPage(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.original_shape = None
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # UI Elements
        self.label_img = QLabel()
        self.label_img.setAlignment(Qt.AlignCenter)
        self.label_img.setStyleSheet("border: 2px dashed #aaa;")
        self.label_img.setMinimumSize(400, 300)
        layout.addWidget(self.label_img)
        
        btn_load = QPushButton('Load Image')
        btn_load.clicked.connect(self.load_image)
        layout.addWidget(btn_load)
        
        self.label_method = QLabel('Compression Method:')
        layout.addWidget(self.label_method)
        
        self.option_quant = QRadioButton('Median Quantization (32 levels)')
        self.option_bitred = QRadioButton('5-bit Reduction')
        self.option_bitred.setChecked(True)
        layout.addWidget(self.option_quant)
        layout.addWidget(self.option_bitred)
        
        btn_process = QPushButton('Compress and Save')
        btn_process.clicked.connect(self.compress_and_save)
        layout.addWidget(btn_process)
        
        self.setLayout(layout)
        
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Image', '', 
            'Images (*.png *.jpg *.jpeg);;All Files (*)')
        
        if path:
            self.image_path = path
            pixmap = QPixmap(path)
            self.label_img.setPixmap(
                pixmap.scaled(400, 300, Qt.KeepAspectRatio))
            
            # Print original pixel samples
            img = np.array(Image.open(path))
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            print("\nOriginal Image Samples (RGB):")
            print_pixel_samples("Red", img[:,:,0])
            print_pixel_samples("Green", img[:,:,1])
            print_pixel_samples("Blue", img[:,:,2])

    def compress_and_save(self):
        if not self.image_path:
            QMessageBox.warning(self, 'Error', 'No image loaded!')
            return
        
        try:
            img = np.array(Image.open(self.image_path))
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            
            # Process based on selected option
            if self.option_quant.isChecked():
                processed = np.stack([
                    apply_median_quantization(img[:,:,0]),
                    apply_median_quantization(img[:,:,1]),
                    apply_median_quantization(img[:,:,2])
                ], axis=-1)
                print("\nAfter Median Quantization:")
            else:
                processed = np.stack([
                    apply_bit_reduction(img[:,:,0]),
                    apply_bit_reduction(img[:,:,1]),
                    apply_bit_reduction(img[:,:,2])
                ], axis=-1)
                print("\nAfter 5-bit Reduction:")
            
            # Print processed samples
            print_pixel_samples("Red", processed[:,:,0])
            print_pixel_samples("Green", processed[:,:,1])
            print_pixel_samples("Blue", processed[:,:,2])
            
            # Pack 5-bit data efficiently
            packed, shape = pack_5bit_array(processed)
            compressed = zlib.compress(packed.tobytes())
            
            # Save options
            path, _ = QFileDialog.getSaveFileName(
                self, 'Save Compressed Data', '', 
                'MAT Files (*.mat);;All Files (*)')
            
            if path:
                if not path.endswith('.mat'):
                    path += '.mat'
                
                savemat(path, {
                    'compressed_data': np.frombuffer(compressed, dtype=np.uint8),
                    'original_shape': shape,
                    'is_quantized': self.option_quant.isChecked()
                }, do_compression=True)
                
                # Calculate stats
                orig_size = img.nbytes
                comp_size = len(compressed)
                ratio = orig_size / comp_size
                
                print(f"\nCompression Results:")
                print(f"Original: {orig_size/1024:.1f} KB")
                print(f"Compressed: {comp_size/1024:.1f} KB")
                print(f"Ratio: {ratio:.1f}x")
                
                QMessageBox.information(
                    self, 'Success', 
                    f'Saved 5-bit compressed data\n'
                    f'Compression ratio: {ratio:.1f}x')
        
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))
            print("Error:", e)

class ScientificDecompressPage(QWidget):
    def __init__(self):
        super().__init__()
        self.mat_data = None
        self.reconstructed_img = None
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # UI Elements
        self.label_img = QLabel()
        self.label_img.setAlignment(Qt.AlignCenter)
        self.label_img.setStyleSheet("border: 2px dashed #aaa;")
        self.label_img.setMinimumSize(400, 300)
        layout.addWidget(self.label_img)
        
        btn_load = QPushButton('Load .mat File')
        btn_load.clicked.connect(self.load_mat_file)
        layout.addWidget(btn_load)
        
        btn_decompress = QPushButton('Decompress and Show')
        btn_decompress.clicked.connect(self.decompress_and_show)
        layout.addWidget(btn_decompress)
        
        btn_save = QPushButton('Save Reconstructed Image')
        btn_save.clicked.connect(self.save_reconstructed_image)
        layout.addWidget(btn_save)
        
        # Info labels
        self.label_info = QLabel('No file loaded')
        self.label_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_info)
        
        self.setLayout(layout)
    
    def load_mat_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open MAT File', '', 
            'MAT Files (*.mat);;All Files (*)')
        
        if path:
            try:
                self.mat_data = loadmat(path)
                self.label_info.setText(f"Loaded: {path.split('/')[-1]}")
                
                # Print basic info
                print("\nLoaded MAT File Info:")
                print(f"Original shape: {self.mat_data['original_shape']}")
                print(f"Compression method: {'Quantized' if self.mat_data['is_quantized'] else '5-bit Reduced'}")
                print(f"Compressed size: {len(self.mat_data['compressed_data'])/1024:.1f} KB")
                
            except Exception as e:
                QMessageBox.critical(self, 'Error', f"Failed to load MAT file:\n{str(e)}")
    
    def decompress_and_show(self):
        if self.mat_data is None:
            QMessageBox.warning(self, 'Error', 'No MAT file loaded!')
            return
        
        try:
            # Decompress the data
            compressed_data = self.mat_data['compressed_data'].tobytes()
            decompressed = zlib.decompress(compressed_data)
            
            # Unpack the 5-bit data
            packed_data = np.frombuffer(decompressed, dtype=np.uint8)
            reconstructed = unpack_5bit_bytes(packed_data, 
                                           self.mat_data['original_shape'][0])
            
            # Convert back to 8-bit based on compression method
            if self.mat_data['is_quantized']:
                # For quantized images, values are already the median values
                self.reconstructed_img = reconstructed.astype(np.uint8)
            else:
                # For 5-bit reduced, shift back to 8-bit range
                self.reconstructed_img = np.left_shift(reconstructed, 3) + 4
            
            # Convert to QImage and display
            height, width, _ = self.reconstructed_img.shape
            bytes_per_line = 3 * width
            qimg = QImage(
                self.reconstructed_img.data, 
                width, height, bytes_per_line, 
                QImage.Format_RGB888
            )
            self.label_img.setPixmap(QPixmap.fromImage(qimg))
            
            # Print reconstruction info
            print("\nReconstruction Info:")
            print(f"Image shape: {self.reconstructed_img.shape}")
            print("Sample pixel values (top-left 5x5):")
            print(self.reconstructed_img[:5, :5, :])
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f"Decompression failed:\n{str(e)}")
    
    def save_reconstructed_image(self):
        if self.reconstructed_img is None:
            QMessageBox.warning(self, 'Error', 'No image to save!')
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Image', '', 
            'PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)')
        
        if path:
            try:
                Image.fromarray(self.reconstructed_img).save(path)
                QMessageBox.information(self, 'Success', f"Image saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, 'Error', f"Failed to save image:\n{str(e)}")


class WelcomeScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        
        # Title
        title = QLabel('Welcome to RBMQ')
        title.setFont(QFont('Arial', 28, QFont.Bold))
        title.setStyleSheet("color: #3b82f6;")
        title.setAlignment(Qt.AlignCenter)
        
        # Subtitle
        subtitle = QLabel('For Efficient Image Compression')
        subtitle.setFont(QFont('Arial', 16))
        subtitle.setStyleSheet("color: #ccd6f6;")
        subtitle.setAlignment(Qt.AlignCenter)
        
        # Team section
        team_label = QLabel('Research and Development by\nAASTU Electrical and Computer Engineering 5th Year Students')
        team_label.setFont(QFont('Arial', 12, QFont.Bold))
        team_label.setStyleSheet("color: #ccd6f6; margin-top: 30px;")
        team_label.setAlignment(Qt.AlignCenter)
        
        # Names
        names = [
            "1. Fikresilase Wondmeneh ",
            "2. Eyuel Mulugeta              ",
            "3. Eyerusalem Desalegn    ",
            "4. Haymanot Sileshi            ",
            "5. Feven Yohanis               "
        ]
        
        names_layout = QVBoxLayout()
        names_layout.setSpacing(8)
        for name in names:
            name_label = QLabel(name)
            name_label.setFont(QFont('Arial', 12))
            name_label.setStyleSheet("color: #ccd6f6;")
            name_label.setAlignment(Qt.AlignCenter)
            names_layout.addWidget(name_label)
        # Advisor label
        advisor_label = QLabel("Advised by: MR. Girma")
        advisor_label.setFont(QFont('Arial', 11, QFont.Bold))
        advisor_label.setStyleSheet("color: #ccd6f6; margin-top: 15px;")

# Wrap the advisor label in a layout to align it properly to the right
        advisor_layout = QHBoxLayout()
        advisor_layout.addStretch()  # Pushes the label to the right
        advisor_layout.addWidget(advisor_label)

        names_layout.addLayout(advisor_layout)

    
        
        # Continue button
        continue_btn = QPushButton('Continue to the App')
        continue_btn.setFont(QFont('Arial', 14))
        continue_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 5px;
                padding: 12px 24px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        continue_btn.setCursor(Qt.PointingHandCursor)
        
        # Add widgets to layout
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch(1)
        layout.addWidget(team_label)
        layout.addLayout(names_layout)
        layout.addStretch(2)
        layout.addWidget(continue_btn, alignment=Qt.AlignCenter)
        layout.addStretch(1)
        
        self.setLayout(layout)
        self.setStyleSheet("background-color: #0a192f;")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('RBMQ Image Processor')
        self.setGeometry(100, 100, 900, 700)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Stacked widget for welcome screen and main app
        self.stacked_widget = QStackedWidget()
        
        # Create welcome screen
        self.welcome_screen = WelcomeScreen()
        self.welcome_screen.findChild(QPushButton).clicked.connect(self.show_main_app)
        
        # Create main app content
        self.main_app_widget = QWidget()
        self.setup_main_app()
        
        # Add to stacked widget
        self.stacked_widget.addWidget(self.welcome_screen)  # Index 0
        self.stacked_widget.addWidget(self.main_app_widget)  # Index 1
        
        # Set welcome screen as initial view
        self.stacked_widget.setCurrentIndex(0)
        
        main_layout.addWidget(self.stacked_widget)
        main_widget.setLayout(main_layout)
        
        # Apply styles
        self.apply_styles()
    
    def setup_main_app(self):
        """Setup the original main application interface"""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        sidebar_layout.setSpacing(20)
        
        # Logo
        logo = QLabel('RBMQ')
        logo.setFont(QFont('Arial', 20, QFont.Bold))
        logo.setStyleSheet("color: #3498db;")
        sidebar_layout.addWidget(logo)
        
        # Day-to-Day Use Section
        day_group = QGroupBox("Day-to-Day Use")
        day_layout = QVBoxLayout()
        
        self.day_compress_btn = QPushButton('Compress')
        self.day_compress_btn.setCheckable(True)
        self.day_compress_btn.setChecked(True)
        
        self.day_decompress_btn = QPushButton('Decompress')
        self.day_decompress_btn.setCheckable(True)
        
        day_layout.addWidget(self.day_compress_btn)
        day_layout.addWidget(self.day_decompress_btn)
        day_group.setLayout(day_layout)
        sidebar_layout.addWidget(day_group)
        
        # Scientific Use Section
        sci_group = QGroupBox("Scientific Use")
        sci_layout = QVBoxLayout()
        
        self.sci_compress_btn = QPushButton('Compress')
        self.sci_compress_btn.setCheckable(True)
        
        self.sci_decompress_btn = QPushButton('Decompress')
        self.sci_decompress_btn.setCheckable(True)
        
        sci_layout.addWidget(self.sci_compress_btn)
        sci_layout.addWidget(self.sci_decompress_btn)
        sci_group.setLayout(sci_layout)
        sidebar_layout.addWidget(sci_group)
        
        sidebar_layout.addStretch()
        sidebar.setLayout(sidebar_layout)
        
        # Content area
        self.content = QStackedWidget()
        
        # Add pages
        self.day_compress_page = DayToDayCompressPage()
        self.day_decompress_page = DayToDayDecompressPage()
        self.sci_compress_page = ScientificCompressPage()
        self.sci_decompress_page = ScientificDecompressPage()
        
        self.content.addWidget(self.day_compress_page)  # Index 0
        self.content.addWidget(self.day_decompress_page)  # Index 1
        self.content.addWidget(self.sci_compress_page)  # Index 2
        self.content.addWidget(self.sci_decompress_page)  # Index 3
        
        # Add to main layout
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.content)
        
        self.main_app_widget.setLayout(main_layout)
        
        # Connect signals
        self.day_compress_btn.clicked.connect(lambda: self.show_page(0))
        self.day_decompress_btn.clicked.connect(lambda: self.show_page(1))
        self.sci_compress_btn.clicked.connect(lambda: self.show_page(2))
        self.sci_decompress_btn.clicked.connect(lambda: self.show_page(3))
    
    def show_main_app(self):
        """Switch from welcome screen to main application"""
        self.stacked_widget.setCurrentIndex(1)
    
    def show_page(self, index):
        self.content.setCurrentIndex(index)
        
        # Update button states
        self.day_compress_btn.setChecked(index == 0)
        self.day_decompress_btn.setChecked(index == 1)
        self.sci_compress_btn.setChecked(index == 2)
        self.sci_decompress_btn.setChecked(index == 3)
    
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a192f;
            }
            QWidget {
                background-color: #0a192f;
                color: #ccd6f6;
            }
            QPushButton {
                background-color: #1e3a8a;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                border: none;
                min-width: 160px;
            }
            QPushButton:hover {
                background-color: #1e40af;
            }
            QPushButton:checked {
                background-color: #3b82f6;
                font-weight: bold;
            }
            QLabel#imageLabel {
                background-color: #112240;
                border: 2px dashed #1e3a8a;
            }
            QRadioButton {
                font-size: 14px;
                spacing: 5px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QMessageBox {
                background-color: #0a192f;
            }
            QMessageBox QLabel {
                color: #ccd6f6;
            }
            QGroupBox {
                border: 1px solid #1e3a8a;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: #ccd6f6;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())