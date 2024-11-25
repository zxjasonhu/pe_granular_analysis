import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from deprecated import deprecated
from matplotlib.widgets import Slider

# RGB color lists, (R, G, B)
colors = [
    (102, 194, 165),  # green
    (252, 141, 98),  # orange
    (141, 160, 203),  # blue
    (231, 138, 195),  # pink
    (166, 216, 84),  # light green
    (255, 217, 47),  # yellow
    (229, 196, 148),  # light brown
    (217, 239, 139),  # light green
    (247, 182, 210),  # light pink
    (253, 180, 98),  # orange
    (205, 151, 151),  # light red
]


def load_dcm_from_dir(path):
    """
    Load dicom files from a directory
    :param path:
    :return: 3D image in numpy array
    """
    dicom_files = os.listdir(path)
    return load_dcm(path, dicom_files)


def load_dcm(path, dicom_files):
    """
    Load dicom files from a directory
    :param path:
    :param dicom_files:
    :return: 3D image in numpy array
    """
    datasets = []
    for file in dicom_files:
        if ".dcm" not in file:
            file = file + ".dcm"
        full_path = os.path.join(path, file)
        ds = pydicom.dcmread(full_path)
        datasets.append(ds)

    datasets.sort(key=lambda x: x.InstanceNumber)
    npa = np.array([ds.pixel_array for ds in datasets])
    return npa


class ImageSliceViewer3D:
    def __init__(
        self,
        volume,
        masks=None,
        pixel_spacing=None,
        slice_thickness=None,
        figsize=(5, 5),
        cmap="gray",
        views=None,
    ):
        """
        Visualize 3D volumes as slices in different planes. Input to the class should be in [C, H, W] format.
        :param volume: Volume should be [C, H, W] format. C is the number of slices. it will be transposed to [H, W, C] to visualize
        :param pixel_spacing: Pixel spacing of the slices. Can be found in dicom files or metadata.json or nifti header
        :param slice_thickness: Slice thickness of the slices. Can be found in dicom files or metadata.json or nifti header
        :param figsize: Figure size for the plots, the larger, the slower the plotting
        :param cmap: Color map to use for displaying the images, default is 'gray'
        :param views: List of views to display, default is ['axial']; options are ['axial', 'coronal', 'sagittal']
        """

        if views is None:
            views = ["axial"]

        self.volume = volume
        if masks is not None:
            assert masks.shape == volume.shape
            # highlight the masked region on volume:
            masked = np.ma.masked_where(masks == 1, masks)
            self.volume = masked.mask * 0.5 + volume * 0.5

        self.figsize = figsize
        self.cmap = cmap
        self.v = [np.min(volume), np.max(volume)]

        # Calculate aspect based on pixel spacing and slice thickness
        if pixel_spacing is not None and slice_thickness is not None:
            self.aspect = float(slice_thickness) / float(pixel_spacing[1])
        else:
            self.aspect = 1

        # Transpose and rotate the volumes to get different views
        # Adjust the aspect attribute according to the orientation

        for view in views:
            if view == "axial":
                self.vol1 = np.transpose(self.volume, [1, 2, 0])  # Axial
                self.f1, self.ax1 = plt.subplots(1, 1, figsize=self.figsize)
                self.axial_image = self.ax1.imshow(
                    self.vol1[:, :, 0],
                    cmap=plt.get_cmap(self.cmap),
                    vmin=self.v[0],
                    vmax=self.v[1],
                )
                self.slider_ax1 = self.f1.add_axes([0.2, 0.02, 0.65, 0.03])
                self.slider1 = Slider(
                    self.slider_ax1,
                    "Axial",
                    0,
                    self.vol1.shape[2] - 1,
                    valinit=0,
                    valstep=1,
                )
                self.slider1.on_changed(self.plot_slice1)

            elif view == "coronal":
                self.vol2 = np.rot90(np.transpose(self.volume, [2, 0, 1]), 3)  # Coronal
                self.f2, self.ax2 = plt.subplots(1, 1, figsize=self.figsize)
                self.coronal_image = self.ax2.imshow(
                    self.vol2[:, :, 0],
                    cmap=plt.get_cmap(self.cmap),
                    vmin=self.v[0],
                    vmax=self.v[1],
                    aspect=self.aspect,
                )
                self.slider_ax2 = self.f2.add_axes([0.2, 0.02, 0.65, 0.03])
                self.slider2 = Slider(
                    self.slider_ax2,
                    "Coronal",
                    0,
                    self.vol2.shape[2] - 1,
                    valinit=0,
                    valstep=1,
                )
                self.slider2.on_changed(self.plot_slice2)

            elif view == "sagittal":
                self.vol3 = np.transpose(
                    self.volume, [0, 1, 2]
                )  # Sagittal, this one needs aspect adjustment
                self.f3, self.ax3 = plt.subplots(1, 1, figsize=self.figsize)
                self.sagittal_image = self.ax3.imshow(
                    self.vol3[:, :, 0],
                    cmap=plt.get_cmap(self.cmap),
                    vmin=self.v[0],
                    vmax=self.v[1],
                    aspect=self.aspect,
                )  # aspect adjustment here
                self.slider_ax3 = self.f3.add_axes([0.2, 0.02, 0.65, 0.03])
                self.slider3 = Slider(
                    self.slider_ax3,
                    "Sagittal",
                    0,
                    self.vol3.shape[2] - 1,
                    valinit=0,
                    valstep=1,
                )
                self.slider3.on_changed(self.plot_slice3)

            else:
                raise ValueError("View type not supported")

        # Adjust layout
        # self.f1.tight_layout()
        # self.f2.tight_layout()
        # self.f3.tight_layout()

    def plot_slice1(self, z1):
        # Update the images
        self.axial_image.set_data(self.vol1[:, :, int(z1)])
        self.f1.canvas.draw_idle()
        self.f1.canvas.flush_events()

    def plot_slice2(self, z2):
        # Update the images
        self.coronal_image.set_data(self.vol2[:, :, int(z2)])
        self.f2.canvas.draw_idle()
        self.f2.canvas.flush_events()

    def plot_slice3(self, z3):
        # Update the images
        self.sagittal_image.set_data(self.vol3[:, :, int(z3)])
        self.f3.canvas.draw_idle()
        self.f3.canvas.flush_events()


class ColorImageSliceViewer3D:
    def __init__(
        self,
        volume,
        masks=None,
        bboxes=None,
        pixel_spacing=None,
        slice_thickness=None,
        figsize=(5, 5),
        cmap="jet",
        views=None,
    ):
        """
        Visualize 3D volumes as slices in different planes. Input to the class should be in [L, H, W] format.
        :param volume: Volume should be [L, H, W] format. L is the number of slices. it will be transposed to [C, H, W, L] to visualize
        :param pixel_spacing: Pixel spacing of the slices. Can be found in dicom files or metadata.json or nifti header
        :param slice_thickness: Slice thickness of the slices. Can be found in dicom files or metadata.json or nifti header
        :param figsize: Figure size for the plots, the larger, the slower the plotting
        :param cmap: Color map to use for displaying the images, default is 'gray'
        :param views: List of views to display, default is ['axial']; options are ['axial', 'coronal', 'sagittal']
        """

        if views is None:
            views = ["axial"]

        if volume.dtype == np.uint8:
            volume = volume.astype(np.float32) / 255.0

        self.volume = volume
        self.volume -= np.min(self.volume)
        self.volume /= np.max(self.volume)
        # convert to 3 channel C, H, W, L
        if len(self.volume.shape) == 3:
            self.volume = np.repeat(
                self.volume[..., np.newaxis], 3, axis=-1
            )  # [L, H, W, C]

        self.cmap = cmap

        if masks is not None:
            # highlight the masked region on volume:
            for ind, m in enumerate(masks):
                assert m.shape == volume.shape
                # apply different colormap for each mask
                for i in range(3):
                    self.volume[:, :, :, i] += m * colors[ind][i] / 255

        if bboxes is not None:
            for ind, bbox in enumerate(bboxes):
                xmin, ymin, zmin, xmax, ymax, zmax = bbox
                # draw the bounding box
                for i in range(3):
                    self.volume[zmin:zmax, ymin, xmin:xmax, i] = colors[ind][i] / 255
                    self.volume[zmin:zmax, ymax, xmin:xmax, i] = colors[ind][i] / 255
                    self.volume[zmin:zmax, ymin:ymax, xmin, i] = colors[ind][i] / 255
                    self.volume[zmin:zmax, ymin:ymax, xmax, i] = colors[ind][i] / 255

        self.volume -= np.min(self.volume)
        self.volume /= np.max(self.volume)
        # normalize the volume
        self.v = [np.min(volume), np.max(volume)]
        self.figsize = figsize

        # Calculate aspect based on pixel spacing and slice thickness
        if pixel_spacing is not None and slice_thickness is not None:
            self.aspect = float(slice_thickness) / float(pixel_spacing[1])
        else:
            self.aspect = 1

        # Transpose and rotate the volumes to get different views
        # Adjust the aspect attribute according to the orientation

        for view in views:
            if view == "axial":
                self.vol1 = np.transpose(self.volume, [0, 1, 2, 3])
                print(self.vol1.shape)
                self.f1, self.ax1 = plt.subplots(1, 1, figsize=self.figsize)
                self.axial_image = self.ax1.imshow(
                    self.vol1[
                        0,
                        :,
                        :,
                        :,
                    ],
                )
                self.slider_ax1 = self.f1.add_axes([0.2, 0.02, 0.65, 0.03])
                self.slider1 = Slider(
                    self.slider_ax1,
                    "Axial",
                    0,
                    self.vol1.shape[0] - 1,
                    valinit=0,
                    valstep=1,
                )
                self.slider1.on_changed(self.plot_slice1)

            elif view == "coronal":
                self.vol2 = np.rot90(np.transpose(self.volume, [2, 0, 1, 3]), 3)
                self.f2, self.ax2 = plt.subplots(1, 1, figsize=self.figsize)
                self.coronal_image = self.ax2.imshow(
                    self.vol2[0, :, :, :],
                    aspect=self.aspect,
                )
                self.slider_ax2 = self.f2.add_axes([0.2, 0.02, 0.65, 0.03])
                self.slider2 = Slider(
                    self.slider_ax2,
                    "Coronal",
                    0,
                    self.vol2.shape[2] - 1,
                    valinit=0,
                    valstep=1,
                )
                self.slider2.on_changed(self.plot_slice2)

            elif view == "sagittal":
                self.vol3 = np.transpose(
                    self.volume, [1, 2, 3, 0]
                )  # Sagittal, this one needs aspect adjustment
                self.f3, self.ax3 = plt.subplots(1, 1, figsize=self.figsize)
                self.sagittal_image = self.ax3.imshow(
                    self.vol3[0, :, :, :],
                    aspect=self.aspect,
                )  # aspect adjustment here
                self.slider_ax3 = self.f3.add_axes([0.2, 0.02, 0.65, 0.03])
                self.slider3 = Slider(
                    self.slider_ax3,
                    "Sagittal",
                    0,
                    self.vol3.shape[2] - 1,
                    valinit=0,
                    valstep=1,
                )
                self.slider3.on_changed(self.plot_slice3)

            else:
                raise ValueError("View type not supported")

        # Adjust layout
        # self.f1.tight_layout()
        # self.f2.tight_layout()
        # self.f3.tight_layout()

    def plot_slice1(self, z1):
        # Update the images
        self.axial_image.set_data(
            self.vol1[
                int(z1),
                :,
                :,
                :,
            ]
        )
        self.f1.canvas.draw_idle()
        self.f1.canvas.flush_events()

    def plot_slice2(self, z2):
        # Update the images
        self.coronal_image.set_data(self.vol2[int(z2), :, :, :])
        self.f2.canvas.draw_idle()
        self.f2.canvas.flush_events()

    def plot_slice3(self, z3):
        # Update the images
        self.sagittal_image.set_data(self.vol3[int(z3), :, :, :])
        self.f3.canvas.draw_idle()
        self.f3.canvas.flush_events()


@deprecated(reason="Use ImageSliceViewer3D instead", version="0.1.0")
class CTViewer3D:
    def __init__(
        self, volume, pixel_spacing, slice_thickness, figsize=(5, 5), cmap="gray"
    ):
        """
        Visualize 3D volumes as slices in different planes. Input to the class should be in [C, H, W] format.
        :param volume: Volume should be [C, H, W] format. C is the number of slices. it will be transposed to [H, W, C] to visualize
        :param pixel_spacing: Pixel spacing of the slices. Can be found in dicom files or metadata.json or nifti header
        :param slice_thickness: Slice thickness of the slices. Can be found in dicom files or metadata.json or nifti header
        :param figsize: Figure size for the plots, the larger, the slower the plotting
        :param cmap: Color map to use for displaying the images, default is 'gray'
        :param views: visualize everything by default
        """

        self.volume = volume
        self.figsize = figsize
        self.cmap = cmap
        self.v = [np.min(volume), np.max(volume)]

        # Calculate aspect based on pixel spacing and slice thickness
        self.aspect = float(slice_thickness) / float(pixel_spacing[1])

        # Transpose and rotate the volumes to get different views
        # Adjust the aspect attribute according to the orientation
        self.vol1 = np.transpose(self.volume, [1, 2, 0])  # Axial
        self.vol2 = np.rot90(np.transpose(self.volume, [2, 0, 1]), 3)  # Coronal
        self.vol3 = np.transpose(
            self.volume, [0, 1, 2]
        )  # Sagittal, this one needs aspect adjustment

        self.f1, self.ax1 = plt.subplots(1, 1, figsize=self.figsize)
        self.f2, self.ax2 = plt.subplots(1, 1, figsize=self.figsize)
        self.f3, self.ax3 = plt.subplots(1, 1, figsize=self.figsize)

        # Initialize the images
        self.images = [
            self.ax1.imshow(
                self.vol1[:, :, 0],
                cmap=plt.get_cmap(self.cmap),
                vmin=self.v[0],
                vmax=self.v[1],
            ),
            self.ax2.imshow(
                self.vol2[:, :, 0],
                cmap=plt.get_cmap(self.cmap),
                vmin=self.v[0],
                vmax=self.v[1],
                aspect=self.aspect,
            ),
            self.ax3.imshow(
                self.vol3[:, :, 0],
                cmap=plt.get_cmap(self.cmap),
                vmin=self.v[0],
                vmax=self.v[1],
                aspect=self.aspect,
            ),  # aspect adjustment here
        ]

        # Create slider axes
        self.slider_ax1 = self.f1.add_axes([0.2, 0.02, 0.65, 0.03])
        self.slider_ax2 = self.f2.add_axes([0.2, 0.02, 0.65, 0.03])
        self.slider_ax3 = self.f3.add_axes([0.2, 0.02, 0.65, 0.03])

        # Create sliders
        self.slider1 = Slider(
            self.slider_ax1, "Axial", 0, self.vol1.shape[2] - 1, valinit=0, valstep=1
        )
        self.slider2 = Slider(
            self.slider_ax2, "Coronal", 0, self.vol2.shape[2] - 1, valinit=0, valstep=1
        )
        self.slider3 = Slider(
            self.slider_ax3, "Sagittal", 0, self.vol3.shape[2] - 1, valinit=0, valstep=1
        )

        # Call to select slice plane
        self.slider1.on_changed(self.plot_slice1)
        self.slider2.on_changed(self.plot_slice2)
        self.slider3.on_changed(self.plot_slice3)

        # Adjust layout
        # self.f1.tight_layout()
        # self.f2.tight_layout()
        # self.f3.tight_layout()

    def plot_slice1(self, z1):
        # Update the images
        self.images[0].set_data(self.vol1[:, :, int(z1)])
        self.f1.canvas.draw_idle()
        self.f1.canvas.flush_events()

    def plot_slice2(self, z2):
        # Update the images
        self.images[1].set_data(self.vol2[:, :, int(z2)])
        self.f2.canvas.draw_idle()
        self.f2.canvas.flush_events()

    def plot_slice3(self, z3):
        # Update the images
        self.images[2].set_data(self.vol3[:, :, int(z3)])
        self.f3.canvas.draw_idle()
        self.f3.canvas.flush_events()
