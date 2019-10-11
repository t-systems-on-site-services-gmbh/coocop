import numpy as np
import pytest
import coocop


def test_Copyout_internal_vars():
    coo = coocop.Copyout(77, 23)
    assert coo._extent == 77
    assert coo._image_buffer_size == 23

def test_Copyout_call():
    coo = coocop.Copyout(11, 13)
    input_image = np.random.rand(31, 33, 3)
    output_image = coo(input_image)
    output_image_shape = output_image.shape
    input_image_shape = input_image.shape
    assert input_image_shape == output_image_shape

def test_Copyout_buffer_first():
    coo = coocop.Copyout(11, 13)
    input_image = np.random.rand(31, 33, 3)
    coo(input_image)
    assert len(coo._image_buffer) == 1

def test_Copyout_buffer_full():
    BUFFER_LEN = 3
    coo = coocop.Copyout(11, BUFFER_LEN)
    for _ in range(BUFFER_LEN+10):
        input_image = np.random.rand(31, 33, 3)
        coo(input_image)
    assert len(coo._image_buffer) == BUFFER_LEN

def test_CopyPairing_on_epoch_begin():
    EPOCH_NUMBER = 77
    cop = coocop.CopyPairing(
            extent = 16,
            warmup_epochs = 100,
            fine_tuning_epoch = 300,
            coo_epochs = 1,
            cop_epochs = 1,
            )
    cop.on_epoch_begin(EPOCH_NUMBER)
    assert cop._current_epoch == EPOCH_NUMBER

def test_CopyPairing_call():
    cop = coocop.CopyPairing(
            extent = 16,
            warmup_epochs = 100,
            fine_tuning_epoch = 300,
            coo_epochs = 1,
            cop_epochs = 1,
            )
    input_image = np.random.rand(31, 33, 3)
    output_image = cop(input_image)
    output_image_shape = output_image.shape
    input_image_shape = input_image.shape
    assert input_image_shape == output_image_shape

def test_CopyPairing_buffer_first():
    cop = coocop.CopyPairing(
            extent = 16,
            warmup_epochs = 100,
            fine_tuning_epoch = 300,
            coo_epochs = 1,
            cop_epochs = 1,
            )
    input_image = np.random.rand(31, 33, 3)
    cop(input_image)
    assert len(cop._image_buffer) == 1

def test_CopyPairing_buffer_full():
    BUFFER_LEN = 3
    cop = coocop.CopyPairing(
            extent = 16,
            warmup_epochs = 100,
            fine_tuning_epoch = 300,
            coo_epochs = 1,
            cop_epochs = 1,
            image_buffer_size = BUFFER_LEN,
            )
    for _ in range(BUFFER_LEN+10):
        input_image = np.random.rand(31, 33, 3)
        cop(input_image)
    assert len(cop._image_buffer) == BUFFER_LEN
