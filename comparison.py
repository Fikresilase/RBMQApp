import sys
import os
import numpy as np
from PIL import Image
import pandas as pd
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
        
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Header
        header = QLabel("Image Quality Analysis Dashboard")
        header.setStyleSheet("""
            font-size: 24px; 
            font-weight: bold; 
            color: #2c3e50;
            padding: 10px;
        """)
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Select Image Folder")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                font-size: 14px; 
                border-radius: 4px;
            }
            QPushButton:hover { 
                background-color: #2980b9; 
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
                padding: 8px; 
                font-size: 14px; 
                border-radius: 4px; 
                border: 1px solid #bdc3c7; 
                min-width: 200px;
            }
        """)
        control_layout.addWidget(QLabel("Display Mode:"))
        control_layout.addWidget(self.metric_combo)
        
        control_layout.addStretch()
        main_layout.addLayout(control_layout)
        
        # Reference image info
        self.reference_info = QLabel("No reference image loaded")
        self.reference_info.setStyleSheet("font-style: italic; color: #7f8c8d;")
        self.reference_info.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.reference_info)
        
        # Results area (tabs)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab { 
                padding: 8px 12px; 
                font-size: 14px; 
            }
            QTabWidget::pane { 
                border: 1px solid #bdc3c7; 
            }
        """)
        main_layout.addWidget(self.tabs)
        
        # Table tab
        self.table_tab = QWidget()
        self.table_layout = QVBoxLayout()
        self.table_tab.setLayout(self.table_layout)
        
        self.results_table = QTableWidget()
        self.results_table.setStyleSheet("""
            QTableWidget { 
                font-size: 12px; 
                selection-background-color: #d6eaf8;
            }
            QHeaderView::section { 
                background-color: #ecf0f1; 
                padding: 6px; 
                font-weight: bold;
            }
        """)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSortingEnabled(True)
        self.table_layout.addWidget(self.results_table)
        
        # Graph tab
        self.graph_tab = QWidget()
        self.graph_layout = QVBoxLayout()
        self.graph_tab.setLayout(self.graph_layout)
        
        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground('w')
        self.graph_widget.showGrid(x=True, y=True, alpha=0.3)
        self.graph_layout.addWidget(self.graph_widget)
        
        # Thumbnail tab
        self.thumbnail_tab = QWidget()
        self.thumbnail_layout = QVBoxLayout()
        self.thumbnail_tab.setLayout(self.thumbnail_layout)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.thumbnail_scroll_layout = QVBoxLayout(scroll_content)
        
        self.thumbnail_container = QWidget()
        self.thumbnail_container_layout = QVBoxLayout()
        self.thumbnail_container.setLayout(self.thumbnail_container_layout)
        
        self.thumbnail_scroll_layout.addWidget(self.thumbnail_container)
        self.thumbnail_scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        self.thumbnail_layout.addWidget(scroll)
        
        self.tabs.addTab(self.table_tab, "Metrics Table")
        self.tabs.addTab(self.graph_tab, "Visual Comparison")
        self.tabs.addTab(self.thumbnail_tab, "Image Gallery")
    
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
        
        # Configure graph appearance
        self.graph_widget.setLabel('left', 'Metric Value')
        self.graph_widget.setLabel('bottom', 'Images')
        self.graph_widget.getPlotItem().getAxis('bottom').setHeight(150)
        self.graph_widget.getPlotItem().getAxis('left').setWidth(80)
    
    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder_path:
            self.process_folder(folder_path)
    
    def process_folder(self, folder_path):
        self.image_data = []
        self.bmp_reference = None
        
        # Find the first image to use as reference (prefer BMP if available)
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not image_files:
            self.reference_info.setText("⚠️ No images found in folder")
            return
        
        # Try to find a BMP first, otherwise use the first image
        for f in image_files:
            if f.lower().endswith('.bmp'):
                self.bmp_reference = os.path.join(folder_path, f)
                break
        else:
            self.bmp_reference = os.path.join(folder_path, image_files[0])
        
        # Load reference image
        try:
            with Image.open(self.bmp_reference) as ref_img:
                ref_array = np.array(ref_img.convert('RGB'))
                
                # Store reference dimensions for comparison
                ref_width, ref_height = ref_img.size
                
                self.reference_info.setText(
                    f"Reference: {os.path.basename(self.bmp_reference)} | "
                    f"Size: {ref_width}×{ref_height} | "
                    f"File Size: {os.path.getsize(self.bmp_reference)/1024:.1f} KB"
                )
                
                # Process all other images in folder
                for filename in image_files:
                    if filename == os.path.basename(self.bmp_reference):
                        continue
                        
                    filepath = os.path.join(folder_path, filename)
                    try:
                        with Image.open(filepath) as img:
                            img_array = np.array(img.convert('RGB'))
                            img_width, img_height = img.size
                            
                            # Calculate metrics without resizing
                            if ref_array.shape[:2] != img_array.shape[:2]:
                                # Note the size difference but still calculate metrics
                                size_note = f" (Size diff: {ref_width}x{ref_height} vs {img_width}x{img_height})"
                            else:
                                size_note = ""
                            
                            # Convert to LAB color space for Delta E
                            ref_lab = color.rgb2lab(ref_array / 255.0)
                            img_lab = color.rgb2lab(img_array / 255.0)
                            
                            # Calculate metrics that don't require same dimensions
                            delta_e = color.deltaE_ciede2000(ref_lab, img_lab)
                            entropy = shannon_entropy(img_array)
                            file_size = os.path.getsize(filepath) / 1024
                            bpp = (os.path.getsize(filepath) * 8) / (img_width * img_height)
                            
                            # Calculate metrics that require same dimensions (with padding if needed)
                            if ref_array.shape == img_array.shape:
                                ssim_value = ssim(ref_array, img_array, channel_axis=2, 
                                                 data_range=img_array.max() - img_array.min())
                                psnr_value = psnr(ref_array, img_array, 
                                                 data_range=img_array.max() - img_array.min())
                                mse_value = mse(ref_array, img_array)
                            else:
                                # Handle size mismatch by padding or other strategy
                                ssim_value = -1  # Indicate invalid comparison
                                psnr_value = -1
                                mse_value = -1
                            
                            self.image_data.append({
                                'Filename': filename + size_note,
                                'Original Filename': filename,
                                'File Size (KB)': file_size,
                                'Compression Ratio': os.path.getsize(self.bmp_reference) / os.path.getsize(filepath),
                                'Delta E (CIEDE2000)': np.mean(delta_e),
                                'Entropy': entropy,
                                'Bit Per Pixel': bpp,
                                'SSIM': ssim_value if ssim_value != -1 else float('nan'),
                                'PSNR': psnr_value if psnr_value != -1 else float('nan'),
                                'MSE': mse_value if mse_value != -1 else float('nan'),
                                'Width': img_width,
                                'Height': img_height
                            })
                            
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                
                # Update UI
                self.display_results_table()
                self.update_graph()
                self.display_thumbnails(folder_path)
            
        except Exception as e:
            self.reference_info.setText(f"⚠️ Error loading reference image: {str(e)}")
    
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
            brush=self.metric_colors[metric],
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
        
        # Select metrics to display
        metrics_to_show = [
            'Compression Ratio',
            'PSNR',
            'SSIM',
            'Delta E (CIEDE2000)',
            'Bit Per Pixel'
        ]
        
        filenames = [img['Original Filename'] for img in self.image_data]
        num_images = len(filenames)
        num_metrics = len(metrics_to_show)
        
        # Create positions for grouped bars
        x = np.arange(num_images)
        width = 0.8 / num_metrics  # Width of each bar group
        
        # Add bars for each metric
        bars = []
        for i, metric in enumerate(metrics_to_show):
            values = [img.get(metric, 0) for img in self.image_data]
            
            # Handle special cases
            if metric == 'PSNR':
                valid_values = [v for v in values if v != float('inf') and not np.isnan(v)]
                max_val = max(valid_values) if valid_values else 100
                values = [v if v != float('inf') and not np.isnan(v) else max_val * 1.2 for v in values]
            elif metric in ['SSIM', 'Delta E (CIEDE2000)']:
                values = [v if not np.isnan(v) else 0 for v in values]
            
            # Position bars side by side
            x_pos = x + i * width
            
            bar = pg.BarGraphItem(
                x=x_pos,
                height=values,
                width=width,
                brush=self.metric_colors[metric],
                pen=pg.mkPen(color='#2c3e50', width=0.5),
                name=metric
            )
            self.graph_widget.addItem(bar)
            bars.append(bar)
            
            # Add value labels for each bar
            for j, (xj, val) in enumerate(zip(x_pos, values)):
                if not np.isnan(val) and val != float('inf'):
                    text = pg.TextItem(
                        text=f"{val:.2f}",
                        color=(0, 0, 0),
                        anchor=(0.5, 1),
                        angle=90
                    )
                    text.setPos(xj, val)
                    self.graph_widget.addItem(text)
        
        # Customize plot
        self.graph_widget.setLabel('left', 'Metric Values')
        self.graph_widget.setLabel('bottom', 'Images')
        self.graph_widget.setTitle('Multi-Parameter Image Comparison (vs Reference Image)')
        
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
        
        # First show the reference image
        self.add_thumbnail(
            self.bmp_reference, 
            is_reference=True,
            reference_size=os.path.getsize(self.bmp_reference)
        )
        
        # Then show all other images
        for img_data in sorted(self.image_data, key=lambda x: x['Original Filename']):
            filepath = os.path.join(folder_path, img_data['Original Filename'])
            self.add_thumbnail(filepath, img_data)
        
        # Add stretch to push content up
        self.thumbnail_container_layout.addStretch()
    
    def add_thumbnail(self, filepath, img_data=None, is_reference=False, reference_size=None):
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
            if is_reference:
                with Image.open(filepath) as img:
                    width, height = img.size
                dim_label = QLabel(f"Dimensions: {width}×{height}")
            elif img_data:
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
                
                size_mb = reference_size / (1024 * 1024)
                info_layout.addWidget(QLabel(f"Size: {size_mb:.2f} MB"))
            elif img_data:
                # Add key metrics with color coding
                metrics = [
                    ("Compression", f"{img_data['Compression Ratio']:.2f}x", "#e74c3c"),
                    ("PSNR", f"{img_data['PSNR']:.1f} dB" if not np.isnan(img_data['PSNR']) else "N/A", "#d35400"),
                    ("SSIM", f"{img_data['SSIM']:.3f}" if not np.isnan(img_data['SSIM']) else "N/A", "#1abc9c"),
                    ("ΔE", f"{img_data['Delta E (CIEDE2000)']:.2f}", "#2ecc71"),
                    ("Size", f"{img_data['File Size (KB)']:.1f} KB", "#3498db")
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