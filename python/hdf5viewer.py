#!/usr/bin/env python3
import sys
import h5py
import numpy as np
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt

class HDF5TreeWidget(QtWidgets.QTreeWidget):
    def __init__(self, h5file, parent=None):
        super().__init__(parent)
        self.h5file = h5file
        self.setColumnCount(1)
        self.setHeaderLabels(["HDF5 Structure"])
        self.populate_tree()

    def populate_tree(self):
        # Recursively add groups and datasets as tree items.
        def add_items(parent_item, h5node):
            for key in h5node:
                child = h5node[key]
                item = QtWidgets.QTreeWidgetItem([key])
                item.h5node = child  # attach the h5py object to the tree item
                parent_item.addChild(item)
                if isinstance(child, h5py.Group):
                    add_items(item, child)
        add_items(self.invisibleRootItem(), self.h5file)

    def mouseDoubleClickEvent(self, event):
        item = self.currentItem()
        if item and isinstance(item.h5node, h5py.Dataset):
            data = item.h5node[()]
            # Visualize based on dataset dimensionality
            if data.ndim == 1:
                plt.figure()
                plt.plot(data)
                plt.title(f"{item.text(0)} (1D data)")
                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.show()
            elif data.ndim == 2:
                plt.figure()
                plt.imshow(data, aspect='auto', interpolation='none')
                plt.title(f"{item.text(0)} (2D data)")
                plt.colorbar()
                plt.show()
            else:
                QtWidgets.QMessageBox.information(
                    self, "Visualization", "Datasets with >2 dimensions are not supported."
                )
        # Continue with normal behavior.
        super().mouseDoubleClickEvent(event)

class HDF5Viewer(QtWidgets.QMainWindow):
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HDF5 Viewer")
        self.resize(600, 400)
        try:
            self.h5file = h5py.File(file_path, 'r')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open file:\n{e}")
            sys.exit(1)
        self.tree = HDF5TreeWidget(self.h5file)
        self.setCentralWidget(self.tree)

def main():
    # Set the path to your HDF5 file here.
    file_path = 'output/Illustris-3/snapdir_135/combined.hdf5'  
    app = QtWidgets.QApplication(sys.argv)
    viewer = HDF5Viewer(file_path)
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()