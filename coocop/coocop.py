import numpy as np
import tensorflow


class Copyout:
    """
    Copyout class for image augmentation.

    Attributes
    ----------
    extend : int
        Extend of the quare patch. Must be > 0.
    image_buffer_size : int
        Buffer size where images are stores. Must be > 0, default is 128.
    """

    def __init__(self, extent, image_buffer_size=128):
        if not extent > 0:
            raise ValueError('"extend" must be > 0')

        if not image_buffer_size > 0:
            raise ValueError('"image_buffer_size" must be > 0')

        self._extent = extent
        self._image_buffer_size = image_buffer_size
        self._image_buffer = []

    def __call__(self, img):
        """
        Augment a given image with Copyout.

        Attributes
        ----------
        img : numpy tensor with rank 3
            the image

        Returns
        -------
        numpy tensor with rank 3
            the augmented image
        """
        h, w, _ = img.shape

        x = np.random.randint(w)
        y = np.random.randint(h)

        x1 = np.clip(x - self._extent // 2, 0, w)
        x2 = np.clip(x + self._extent // 2, 0, w)
        y1 = np.clip(y - self._extent // 2, 0, h)
        y2 = np.clip(y + self._extent // 2, 0, h)

        copyout_y_size = y2 - y1
        copyout_x_size = x2 - x1
        copyout_y = np.random.randint(h - copyout_y_size)
        copyout_x = np.random.randint(w - copyout_x_size)

        image_buffer_len = len(self._image_buffer)
        img_copy = np.copy(img)

        # only augment when we have images in the buffer
        # first image will not be augmented
        if image_buffer_len > 0:
            image_buffer_index = np.random.randint(image_buffer_len)

            # buffer is full
            if image_buffer_len >= self._image_buffer_size:
                old_img = self._image_buffer.pop(image_buffer_index)

            # buffer still needs to be filled
            else:
                old_img = self._image_buffer[image_buffer_index]

            # do the copying
            img[y1: y2, x1: x2, :] = old_img[copyout_y: copyout_y + copyout_y_size,
                                             copyout_x: copyout_x + copyout_x_size,
                                             :]

        # append source image to buffer
        self._image_buffer.append(img_copy)

        return img


# TODO maybe add SamplePairing and Cutout implementations later


class CopyPairing(tensorflow.keras.callbacks.Callback):

    def __init__(self, extent,
                 warmup_epochs, fine_tuning_epoch,
                 coo_epochs=1, cop_epochs=1,
                 image_buffer_size=128):
        super().__init__()

        if not extent > 0:
            raise ValueError('"extend" must be > 0')

        if not warmup_epochs >= 0:
            raise ValueError('"warmup_epochs" must be >= 0')

        if not fine_tuning_epoch > 0:
            raise ValueError('"fine_tuning_epoch" must be > 0')

        # TODO maybe check if warmup_epochs < fine_tuning_epoch or something

        if not coo_epochs > 0:
            raise ValueError('"coo_epochs" must be > 0')

        if not cop_epochs > 0:
            raise ValueError('"cop_epochs" must be > 0')

        if not image_buffer_size > 0:
            raise ValueError('"image_buffer_size" must be > 0')

        self._extent = extent
        self._warmup_epochs = warmup_epochs
        self._fine_tuning_epoch = fine_tuning_epoch
        self._coo_epochs = coo_epochs
        self._cop_epochs = cop_epochs
        self._image_buffer_size = image_buffer_size

        self._image_buffer = []
        self._coo_count = 0
        self._cop_count = 0
        self._current_epoch = 0

    def copyout(self, img, old_img):
        h, w, _ = img.shape

        x = np.random.randint(w)
        y = np.random.randint(h)

        x1 = np.clip(x - self._extent // 2, 0, w)
        x2 = np.clip(x + self._extent // 2, 0, w)
        y1 = np.clip(y - self._extent // 2, 0, h)
        y2 = np.clip(y + self._extent // 2, 0, h)

        copyout_y_size = y2 - y1
        copyout_x_size = x2 - x1
        copyout_y = np.random.randint(h - copyout_y_size)
        copyout_x = np.random.randint(w - copyout_x_size)

        # do the copying
        img[y1: y2, x1: x2, :] = old_img[copyout_y: copyout_y + copyout_y_size,
                                         copyout_x: copyout_x + copyout_x_size,
                                         :]

        return img

    def __call__(self, img):
        image_buffer_len = len(self._image_buffer)
        img_copy = np.copy(img)

        # only augment when we have images in the buffer
        # first image will not be augmented
        if image_buffer_len > 0:
            image_buffer_index = np.random.randint(image_buffer_len)
            if image_buffer_len >= self._image_buffer_size:
                old_img = self._image_buffer.pop(image_buffer_index)
            else:
                old_img = self._image_buffer[image_buffer_index]
            if self._current_epoch < self._warmup_epochs:
                # Copyout
                img = self.copyout(img, old_img)
            elif self._current_epoch > self._fine_tuning_epoch - 2:
                # Copyout
                img = self.copyout(img, old_img)
            else:
                if self._coo_count <= self._coo_epochs - 1:
                    # Copyout
                    img = self.copyout(img, old_img)
                    self._coo_count += 1
                elif self._cop_count <= self._cop_epochs - 1:
                    # SamplePairing
                    img = np.mean(np.array([img, old_img]), axis=0)

                    self._cop_count += 1
                    if self._cop_count > self._cop_epochs - 1:
                        self._coo_count = 0
                        self._cop_count = 0

        self._image_buffer.append(img_copy)

        return img

    # TODO test if *args is ok
    def on_epoch_begin(self, epoch, *args):
        self._current_epoch = epoch
