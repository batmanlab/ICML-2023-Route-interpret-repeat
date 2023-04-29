import cv2
import numpy as np
from scipy import ndimage


class BoundingBoxGenerator:
    def __init__(self, heatmap, mode="hot", percentile=0.95, min_obj_size=1):
        self.heatmap = heatmap
        self.mode = mode
        self.percentile = percentile
        self.min_obj_size = min_obj_size  # number of pixels in the object

    def get_bbox_pct(self):
        # create quantile mask
        if self.mode == "hot":
            q = np.quantile(self.heatmap, self.percentile)
            mask = self.heatmap > q
        elif self.mode == "cold":
            q = np.quantile(self.heatmap, 1 - self.percentile)
            mask = self.heatmap < q
        else:
            raise Exception("Invalid mode.")

        # label connected pixels in the mask
        label_im, nb_labels = ndimage.label(mask)

        # find the sizes of connected pixels
        sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

        # create labeled image
        mask_size = sizes < self.min_obj_size
        remove_pixel = mask_size[label_im]
        label_im[remove_pixel] = 0
        labels = np.unique(label_im)
        label_im = np.searchsorted(labels, label_im)  # sort objects from large to small

        # generate bounding boxes
        bbox = []
        for l in range(1, len(labels)):
            slice_x, slice_y = ndimage.find_objects(label_im == l)[0]
            if (slice_x.start < slice_x.stop) & (slice_y.start < slice_y.stop):
                b = [slice_y.start, slice_x.start, slice_y.stop, slice_x.stop]
                bbox.append(b)
        return bbox


def deprocess_image(image):
    image = image.cpu().numpy()
    image = np.squeeze(np.transpose(image[0], (1, 2, 0)))
    image = image * np.array((0.229, 0.224, 0.225)) + np.array(
        (0.485, 0.456, 0.406)
    )  # un-normalize
    image = image.clip(0, 1)
    return image


def apply_mask(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
