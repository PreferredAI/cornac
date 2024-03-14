"""
Tests for the TransformersVisionModality class. In order to run this test please
insert url_to_beach1, url_to_beach2, url_to_cat in the get_photos method. Use
your favorite beach and cat photos and check the similarity scores.
"""

# add a checker to make sure all requirements needed in the imports here are really present.
# if they are missing skip the respective test
# If a user wants to un these please run: pip install -r cornac/models/dmrl/requirements.txt

try:
    import torch
    import unittest
    from PIL import Image
    from sentence_transformers import util

    from cornac.models.dmrl.transformer_vision import TransformersVisionModality
    import requests
    run_dmrl_test_funcs = True

except ImportError:
    run_dmrl_test_funcs = False

def skip_test_in_case_of_missing_reqs(test_func):
  test_func.__test__ = run_dmrl_test_funcs  # Mark the test function as (non-)discoverable by unittest
  return test_func


# Please insert valid urls here to two beach photos and one cat photo
beach_urls = ["url_to_beach1",
              "url_to_beach2"]
cat_url = "url_to_cat"

class TestTransformersVisionModality(unittest.TestCase):

    def get_photos(self):

        for i, url in enumerate(beach_urls):
            r = requests.get(url)
            with open(f"beach{i}.jpg", "wb") as f:
                f.write(r.content)

        r = requests.get(cat_url)
        with open("cat.jpg", "wb") as f:
            f.write(r.content)

    def setUp(self):
        self.get_photos()
        beach1 = Image.open("beach0.jpg")
        beach2 = Image.open("beach1.jpg")
        cat = Image.open("cat.jpg")
        self.images = [beach1, beach2, cat]
        self.ids = [0, 1]
        self.modality = TransformersVisionModality(images=self.images, ids=self.ids, preencode=True)

    @skip_test_in_case_of_missing_reqs
    @unittest.skipIf("url_to_beach1" in beach_urls, "Please insert a valid url to download 2 beach and one cat photo")
    def test_transform_image_to_tensor(self):
        """
        Tests that an image is transformed correctly to a tensor
        """
        image_tensor_batch = self.modality.transform_images_to_torch_tensor(self.images)
        assert isinstance(image_tensor_batch, torch.Tensor)
        assert image_tensor_batch.shape[0:2] == torch.Size((3, 3)) # 3 images with 3 channels each
        assert image_tensor_batch.shape[2:] == torch.Size(self.modality.image_size)

    @skip_test_in_case_of_missing_reqs
    @unittest.skipIf("url_to_beach1" in beach_urls, "Please insert a valid url to download 2 beach and one cat photo")
    def test_encode_all_images(self):
        """
        Tests that all images are encoded
        """
        self.modality._encode_images()
        assert isinstance(self.modality.features, torch.Tensor)
        assert self.modality.features.shape[0] == len(self.images)
        assert self.modality.features.shape[1] == 1000

    @skip_test_in_case_of_missing_reqs
    @unittest.skipIf("url_to_beach1" in beach_urls, "Please insert a valid url to download 2 beach and one cat photo")
    def test_encoding_quality(self):
        """
        Test similiarity in latent space between some images
        """
        self.modality._encode_images()
        beach1_beach2_similarity = util.cos_sim(self.modality.features[0], self.modality.features[1])
        assert beach1_beach2_similarity > 0.7

        beach_cat_similarity = util.cos_sim(self.modality.features[0], self.modality.features[2])
        assert beach_cat_similarity < 0.1

        assert beach1_beach2_similarity > beach_cat_similarity
