import sys
import os
import numpy as np
from PIL import Image
import pandas as pd
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage import color
from skimage.measure import shannon_entropy
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QTabWidget, QComboBox, QHeaderView, QSizePolicy, QScrollArea
)
from PyQt5.QtGui import QPixmap, QColor, QFont
from PyQt5.QtCore import Qt, QSize


class ImageComparisonTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Image Quality Analyzer")
        self.setGeometry(100, 100, 1600, 1000)
        
        self.image_data = []
        self.bmp_reference = None
        self.current_metric = "PSNR"
        
        self.initUI()
        self.setupGraphStyles()
    
    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header Label
        header = QLabel("Advanced Image Quality Analyzer")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        """)
        main_layout.addWidget(header)

        # Control Panel
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)

        self.load_btn = QPushButton("📁 Select Image Folder")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db; 
                color: white; 
                border: none; 
                padding: 12px 20px; 
                font-size: 14px; 
                border-radius: 6px;
                min-width: 180px;
                font-weight: 500;
            }
            QPushButton:hover { 
                background-color: #2980b9; 
            }
            QPushButton:pressed {
                background-color: #1c5980;
            }
        """)
        self.load_btn.clicked.connect(self.load_folder)
        control_layout.addWidget(self.load_btn)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems([
            "Single Metric View", 
            "Multi-Parameter Comparison"
        ])
        self.metric_combo.setCurrentText("Multi-Parameter Comparison")
        self.metric_combo.currentTextChanged.connect(self.update_graph)
        self.metric_combo.setStyleSheet("""
            QComboBox {
                padding: 10px 12px;
                font-size: 14px;
                border-radius: 6px;
                border: 1px solid #ccc;
                background-color: #ffffff;
                min-width: 220px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox:focus {
                border: 1px solid #3498db;
                outline: none;
            }
        """)
        control_layout.addWidget(QLabel("Display Mode:"))
        control_layout.addWidget(self.metric_combo)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        # Reference Info Label
        self.reference_info = QLabel("No reference image loaded")
        self.reference_info.setAlignment(Qt.AlignCenter)
        self.reference_info.setStyleSheet("""
            font-size: 14px;
            font-style: italic;
            color: #7f8c8d;
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 6px;
        """)
        main_layout.addWidget(self.reference_info)

        # Tabs Area
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                padding: 12px 20px;
                font-size: 15px;
                font-weight: 500;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 4px;
                background-color: #ecf0f1;
                color: #2c3e50;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
                color: white;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 6px;
                background-color: #ffffff;
            }
        """)
        main_layout.addWidget(self.tabs)

        # Table Tab
        self.table_tab = QWidget()
        self.table_layout = QVBoxLayout(self.table_tab)
        self.table_layout.setContentsMargins(10, 10, 10, 10)

        self.results_table = QTableWidget()
        self.results_table.setStyleSheet("""
            QTableWidget {
                font-size: 13px;
                border: 1px solid #ddd;
                border-radius: 6px;
                background-color: #ffffff;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QTableWidget::item:selected {
                background-color: #d6eaf8;
            }
            QHeaderView::section {
                background-color: #f1f3f5;
                padding: 8px 10px;
                font-weight: bold;
                color: #2c3e50;
                border-bottom: 1px solid #ccc;
            }
        """)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSortingEnabled(True)
        self.table_layout.addWidget(self.results_table)

        # Graph Tab
        self.graph_tab = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_tab)
        self.graph_layout.setContentsMargins(10, 10, 10, 10)

        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground('w')
        self.graph_widget.showGrid(x=True, y=True, alpha=0.3)
        self.graph_widget.setTitle("<span style='font-size: 16pt; color: #2c3e50;'>Image Quality Metrics</span>")
        self.graph_layout.addWidget(self.graph_widget)

        # Thumbnail Tab
        self.thumbnail_tab = QWidget()
        self.thumbnail_layout = QVBoxLayout(self.thumbnail_tab)
        self.thumbnail_layout.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f8f9fa;
            }
            QScrollBar:vertical {
                width: 12px;
                background-color: #f1f3f5;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #d0d6de;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background-color: transparent;
            }
        """)

        scroll_content = QWidget()
        self.thumbnail_scroll_layout = QVBoxLayout(scroll_content)
        self.thumbnail_container = QWidget()
        self.thumbnail_container_layout = QVBoxLayout(self.thumbnail_container)
        self.thumbnail_container.setLayout(self.thumbnail_container_layout)
        self.thumbnail_scroll_layout.addWidget(self.thumbnail_container)
        self.thumbnail_scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        self.thumbnail_layout.addWidget(scroll)

        # Add Tabs
        self.tabs.addTab(self.table_tab, "📊 Metrics Table")
        self.tabs.addTab(self.graph_tab, "📈 Visual Comparison")
        self.tabs.addTab(self.thumbnail_tab, "🖼️ Image Gallery")
        
    
    def setupGraphStyles(self):
        self.metric_colors = {
            'File Size (KB)': '#3498db',
            'Compression Ratio': '#e74c3c',
            'Delta E (CIEDE2000)': '#2ecc71',
            'Entropy': '#f39c12',
            'Bit Per Pixel': '#9b59b6',
            'SSIM': '#1abc9c',
            'PSNR': '#d35400',
            'MSE': '#34495e'
        }
        
        self.metric_units = {
            'File Size (KB)': 'KB',
            'Compression Ratio': 'Ratio',
            'Delta E (CIEDE2000)': 'ΔE',
            'Entropy': 'Bits',
            'Bit Per Pixel': 'bpp',
            'SSIM': 'Index (0-1)',
            'PSNR': 'dB',
            'MSE': 'Error'
        }
    
    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder_path:
            self.process_folder(folder_path)
    
    def process_folder(self, folder_path):
        self.image_data = []
        self.bmp_reference = None
        
        # Find all supported files
        supported_files = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.mat'))]
        
        if not supported_files:
            self.reference_info.setText("⚠️ No supported files found in folder")
            return
        
        # Process each file
        for filename in supported_files:
            filepath = os.path.join(folder_path, filename)
            try:
                if filename.lower().endswith('.mat'):
                    # Handle .mat file
                    mat_data = loadmat(filepath)
                    img_array = None
                    
                    # Try common image variable names
                    for var_name in ['image', 'img', 'data', 'X']:
                        if var_name in mat_data:
                            img_array = mat_data[var_name]
                            break
                    
                    # If not found, look for first suitable array
                    if img_array is None:
                        for key, value in mat_data.items():
                            if isinstance(value, np.ndarray) and len(value.shape) >= 2:
                                img_array = value
                                break
                    
                    if img_array is None:
                        continue
                        
                    # Convert to uint8 if needed
                    if img_array.dtype != np.uint8:
                        img_array = (255 * (img_array - img_array.min()) / 
                                   (img_array.max() - img_array.min())).astype(np.uint8)
                    
                    # Convert to 3 channels if grayscale
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array]*3, axis=-1)
                    elif img_array.shape[2] == 1:
                        img_array = np.concatenate([img_array]*3, axis=-1)
                    
                    img = Image.fromarray(img_array)
                    file_size = os.path.getsize(filepath) / 1024
                    is_reference = False
                    
                else:
                    # Handle regular image files
                    img = Image.open(filepath)
                    img_array = np.array(img.convert('RGB'))
                    file_size = os.path.getsize(filepath) / 1024
                    is_reference = filename.lower().endswith('.bmp')
                
                # Set as reference if it's the first BMP found
                if is_reference and self.bmp_reference is None:
                    self.bmp_reference = {
                        'path': filepath,
                        'array': img_array,
                        'image': img,
                        'size': file_size,
                        'width': img.width,
                        'height': img.height,
                        'entropy': shannon_entropy(img_array)
                    }
                    continue
                
                # Calculate metrics
                metrics = {
                    'Filename': filename,
                    'File Size (KB)': file_size,
                    'Width': img.width,
                    'Height': img.height,
                    'Entropy': shannon_entropy(img_array)
                }
                
                if self.bmp_reference:
                    # Calculate comparison metrics
                    ref_array = self.bmp_reference['array']
                    
                    # Resize if dimensions don't match
                    if img_array.shape != ref_array.shape:
                        img = img.resize((self.bmp_reference['width'], self.bmp_reference['height']))
                        img_array = np.array(img.convert('RGB'))
                    
                    metrics.update({
                        'Compression Ratio': self.bmp_reference['size'] / file_size,
                        'Delta E (CIEDE2000)': np.mean(color.deltaE_ciede2000(
                            color.rgb2lab(ref_array/255.0),
                            color.rgb2lab(img_array/255.0)
                        )),
                        'SSIM': ssim(ref_array, img_array, channel_axis=2,
                                    data_range=img_array.max()-img_array.min()),
                        'PSNR': psnr(ref_array, img_array,
                                    data_range=img_array.max()-img_array.min()),
                        'MSE': mse(ref_array, img_array)
                    })
                
                self.image_data.append(metrics)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        
        # Add reference BMP to comparison data if it exists
        if self.bmp_reference:
            self.image_data.insert(0, {
                'Filename': os.path.basename(self.bmp_reference['path']) + " (REFERENCE)",
                'File Size (KB)': self.bmp_reference['size'],
                'Width': self.bmp_reference['width'],
                'Height': self.bmp_reference['height'],
                'Entropy': self.bmp_reference['entropy'],
                'Compression Ratio': 1.0,
                'Delta E (CIEDE2000)': 0.0,
                'SSIM': 1.0,
                'PSNR': float('inf'),
                'MSE': 0.0
            })
        
            self.reference_info.setText(
                f"Reference: {os.path.basename(self.bmp_reference['path'])} | "
                f"Size: {self.bmp_reference['width']}×{self.bmp_reference['height']} | "
                f"File Size: {self.bmp_reference['size']:.1f} KB"
            )
        
        # Update UI
        self.display_results_table()
        self.update_graph()
        self.display_thumbnails(folder_path)
    
    def display_results_table(self):
        if not self.image_data:
            return
        
        # Create DataFrame from image data
        metrics = [
            'File Size (KB)', 'Compression Ratio', 'Delta E (CIEDE2000)',
            'Entropy', 'Bit Per Pixel', 'SSIM', 'PSNR', 'MSE', 'Width', 'Height'
        ]
        
        # Set up table
        self.results_table.setRowCount(len(self.image_data))
        self.results_table.setColumnCount(len(metrics) + 1)  # +1 for filename
        
        headers = ['Filename'] + metrics
        self.results_table.setHorizontalHeaderLabels(headers)
        
        # Fill table with data
        for row_idx, img_data in enumerate(self.image_data):
            for col_idx, metric in enumerate(headers):
                value = img_data.get(metric, '')
                
                # Format values nicely
                if isinstance(value, float):
                    if metric == 'File Size (KB)':
                        display_value = f"{value:.1f}"
                    elif metric in ['SSIM', 'Compression Ratio']:
                        display_value = f"{value:.3f}" if not np.isnan(value) else "N/A"
                    elif metric == 'PSNR' and (value == float('inf') or np.isnan(value)):
                        display_value = "∞" if value == float('inf') else "N/A"
                    else:
                        display_value = f"{value:.2f}" if not np.isnan(value) else "N/A"
                else:
                    display_value = str(value)
                
                item = QTableWidgetItem(display_value)
                item.setData(Qt.UserRole, value)  # Store raw value for sorting
                
                # Color coding based on metric performance
                self.apply_table_item_coloring(item, metric, value)
                
                self.results_table.setItem(row_idx, col_idx, item)
        
        # Set column widths
        self.results_table.resizeColumnsToContents()
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    
    def apply_table_item_coloring(self, item, metric, value):
        """Apply appropriate coloring to table items based on their metric and value"""
        if metric == 'PSNR' and not np.isnan(value):
            if value != float('inf'):
                # Higher PSNR is better (blue gradient)
                intensity = min(230, max(50, int(230 * (value / 100))))
                item.setBackground(QColor(230 - intensity, 230 - intensity, 255))
        elif metric == 'SSIM' and not np.isnan(value):
            # Higher SSIM is better (green gradient)
            intensity = int(230 * value)
            item.setBackground(QColor(230 - intensity, 255, 230 - intensity))
        elif metric == 'MSE' and not np.isnan(value):
            # Lower MSE is better (red gradient)
            valid_mse = [img['MSE'] for img in self.image_data if not np.isnan(img['MSE'])]
            if valid_mse:
                max_mse = max(valid_mse)
                if max_mse > 0:
                    intensity = int(230 * (value / max_mse))
                    item.setBackground(QColor(255, 230 - intensity, 230 - intensity))
        elif metric == 'Delta E (CIEDE2000)':
            # Lower Delta E is better (green gradient)
            max_de = max(img['Delta E (CIEDE2000)'] for img in self.image_data)
            if max_de > 0:
                intensity = int(230 * (value / max_de))
                item.setBackground(QColor(230 - intensity, 255, 230 - intensity))
        elif metric == 'Compression Ratio':
            # Higher compression ratio is better (green gradient)
            max_cr = max(img['Compression Ratio'] for img in self.image_data)
            if max_cr > 0:
                intensity = int(230 * (value / max_cr))
                item.setBackground(QColor(230 - intensity, 255, 230 - intensity))
    
    def update_graph(self):
        if not self.image_data:
            return
        
        self.graph_widget.clear()
        
        if self.metric_combo.currentText() == "Single Metric View":
            self.show_single_metric_view()
        else:
            self.show_multi_metric_view()
    
    def show_single_metric_view(self):
        # Get the most interesting metric to show by default
        metric = "PSNR"
        filenames = [img['Filename'] for img in self.image_data]
        values = [img.get(metric, 0) for img in self.image_data]
        
        # Handle infinite/nan PSNR
        if metric == 'PSNR':
            valid_values = [v for v in values if v != float('inf') and not np.isnan(v)]
            max_val = max(valid_values) if valid_values else 100
            values = [v if v != float('inf') and not np.isnan(v) else max_val * 1.2 for v in values]
        
        # Create positions for bars
        x = np.arange(len(filenames))
        
        # Create bar graph with value labels
        bg = pg.BarGraphItem(
            x=x, 
            height=values, 
            width=0.6, 
            brush=self.metric_colors.get(metric, '#777777'),
            pen=pg.mkPen(color='#2c3e50', width=1),
            name=metric
        )
        self.graph_widget.addItem(bg)
        
        # Add value labels on top of each bar
        for i, (xi, val) in enumerate(zip(x, values)):
            if not np.isnan(val):
                text = pg.TextItem(
                    text=f"{val:.2f}" if not isinstance(val, str) else val,
                    color=(0, 0, 0),
                    anchor=(0.5, 1)
                )
                text.setPos(xi, val)
                self.graph_widget.addItem(text)
        
        # Customize plot
        self.graph_widget.setLabel('left', f"{metric} ({self.metric_units.get(metric, '')})")
        self.graph_widget.setLabel('bottom', 'Images')
        self.graph_widget.setTitle(f"{metric} Comparison (vs Reference Image)")
        
        # Set x-axis ticks with proper rotation
        ticks = [list(zip(x, [f[:15] + "..." if len(f) > 15 else f for f in filenames]))]
        self.graph_widget.getPlotItem().getAxis('bottom').setTicks(ticks)
        
        # Add legend
        legend = pg.LegendItem(offset=(70, 30))
        legend.addItem(bg, metric)
        legend.setParentItem(self.graph_widget.getPlotItem())
    
    def show_multi_metric_view(self):
        """Show grouped bar chart comparing multiple metrics"""
        if not self.image_data:
            return
        
        # Include the requested metrics plus some key ones
        metrics_to_show = [
            'File Size (KB)',
            'Entropy',
            'MSE',
            'Compression Ratio',
            'PSNR',
            'Delta E (CIEDE2000)',
            'SSIM',
            'Bit Per Pixel'
        ]
        
        filenames = [img['Filename'] for img in self.image_data]
        num_images = len(filenames)
        num_metrics = len(metrics_to_show)
        
        # Create positions for grouped bars
        x = np.arange(num_images)
        width = 0.8 / num_metrics  # Width of each bar group
        
        # Add bars for each metric
        bars = []
        for i, metric in enumerate(metrics_to_show):
            values = []
            for img in self.image_data:
                val = img.get(metric, 0)
                # Handle special cases
                if metric == 'PSNR' and (val == float('inf') or np.isnan(val)):
                    val = 0  # Will be labeled as N/A
                values.append(val)
            
            # Position bars side by side
            x_pos = x + i * width
            
            bar = pg.BarGraphItem(
                x=x_pos,
                height=values,
                width=width,
                brush=self.metric_colors.get(metric, '#777777'),
                pen=pg.mkPen(color='#2c3e50', width=0.5),
                name=metric
            )
            self.graph_widget.addItem(bar)
            bars.append(bar)
            
            # Add value labels for each bar
            for j, (xj, val) in enumerate(zip(x_pos, values)):
                if not np.isnan(val) and val != float('inf'):
                    text = pg.TextItem(
                        text=f"{val:.2f}" if isinstance(val, float) else str(val),
                        color=(0, 0, 0),
                        anchor=(0.5, 1),
                        angle=90,
                        fill=pg.mkColor(255, 255, 255, 200)
                    )
                    text.setPos(xj, val)
                    self.graph_widget.addItem(text)
                elif metric == 'PSNR' and val == float('inf'):
                    text = pg.TextItem(
                        text="∞",
                        color=(0, 0, 0),
                        anchor=(0.5, 1),
                        angle=90,
                        fill=pg.mkColor(255, 255, 255, 200)
                    )
                    text.setPos(xj, max([v for v in values if v != float('inf')], default=100)*1.1)
                    self.graph_widget.addItem(text)
        
        # Customize plot
        self.graph_widget.setLabel('left', 'Metric Values')
        self.graph_widget.setLabel('bottom', 'Images')
        self.graph_widget.setTitle('Multi-Parameter Comparison (File Size, Entropy, MSE, etc.)')
        
        # Set x-axis ticks (centered under groups)
        ticks = [list(zip(x + width * (num_metrics-1)/2, [f[:10] + "..." if len(f) > 10 else f for f in filenames]))]
        self.graph_widget.getPlotItem().getAxis('bottom').setTicks(ticks)
        
        # Add legend
        legend = pg.LegendItem(offset=(70, 30))
        for i, metric in enumerate(metrics_to_show):
            legend.addItem(bars[i], metric)
        legend.setParentItem(self.graph_widget.getPlotItem())
    
    def display_thumbnails(self, folder_path):
        # Clear existing thumbnails
        for i in reversed(range(self.thumbnail_container_layout.count())): 
            self.thumbnail_container_layout.itemAt(i).widget().setParent(None)
        
        # Show all images including reference
        for img_data in self.image_data:
            filename = img_data['Original Filename'] if 'Original Filename' in img_data else img_data['Filename'].replace(" (REFERENCE)", "")
            filepath = os.path.join(folder_path, filename)
            self.add_thumbnail(filepath, img_data, is_reference="(REFERENCE)" in img_data['Filename'])
        
        # Add stretch to push content up
        self.thumbnail_container_layout.addStretch()
    
    def add_thumbnail(self, filepath, img_data=None, is_reference=False):
        try:
            # Create thumbnail widget
            thumb_widget = QWidget()
            thumb_widget.setStyleSheet("""
                QWidget {
                    border: 2px solid #bdc3c7;
                    border-radius: 5px;
                    padding: 5px;
                    background-color: white;
                    margin: 5px;
                }
            """)
            thumb_layout = QHBoxLayout()
            thumb_layout.setContentsMargins(5, 5, 5, 5)
            thumb_widget.setLayout(thumb_layout)
            
            # Load and display thumbnail
            pixmap = QPixmap(filepath)
            thumb_label = QLabel()
            thumb_label.setPixmap(pixmap.scaled(
                QSize(150, 150), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
            thumb_label.setAlignment(Qt.AlignCenter)
            thumb_layout.addWidget(thumb_label)
            
            # Add image info
            info_widget = QWidget()
            info_layout = QVBoxLayout()
            info_widget.setLayout(info_layout)
            
            # File name with size information
            filename = os.path.basename(filepath)
            name_label = QLabel(filename)
            name_label.setStyleSheet("""
                QLabel {
                    font-weight: bold; 
                    color: #2c3e50;
                    font-size: 13px;
                }
            """)
            info_layout.addWidget(name_label)
            
            # Add dimensions
            dim_label = QLabel(f"Dimensions: {img_data['Width']}×{img_data['Height']}")
            dim_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
            info_layout.addWidget(dim_label)
            
            # Add metrics or reference marker
            if is_reference:
                ref_label = QLabel("(REFERENCE IMAGE)")
                ref_label.setStyleSheet("""
                    QLabel {
                        font-weight: bold; 
                        color: #e74c3c;
                        font-size: 12px;
                    }
                """)
                info_layout.addWidget(ref_label)
            
            # Add key metrics with color coding
            metrics = [
                ("File Size", f"{img_data['File Size (KB)']:.1f} KB", "#3498db"),
                ("Entropy", f"{img_data['Entropy']:.2f}", "#f39c12"),
                ("MSE", f"{img_data['MSE']:.2f}" if not np.isnan(img_data['MSE']) else "N/A", "#34495e"),
                ("PSNR", f"{img_data['PSNR']:.1f} dB" if not np.isnan(img_data['PSNR']) else "N/A", "#d35400"),
                ("Compression", f"{img_data['Compression Ratio']:.2f}x", "#e74c3c")
            ]
            
            for metric, value, color in metrics:
                metric_widget = QWidget()
                metric_layout = QHBoxLayout()
                metric_widget.setLayout(metric_layout)
                
                name = QLabel(metric)
                name.setStyleSheet(f"color: {color}; font-weight: bold; min-width: 70px; font-size: 11px;")
                metric_layout.addWidget(name)
                
                val = QLabel(value)
                val.setStyleSheet("font-family: monospace; font-size: 11px;")
                metric_layout.addWidget(val)
                
                info_layout.addWidget(metric_widget)
            
            thumb_layout.addWidget(info_widget)
            self.thumbnail_container_layout.addWidget(thumb_widget)
            
            # Add separator
            separator = QWidget()
            separator.setFixedHeight(5)
            self.thumbnail_container_layout.addWidget(separator)
            
        except Exception as e:
            print(f"Error creating thumbnail for {filepath}: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ImageComparisonTool()
    window.show()
    sys.exit(app.exec_())