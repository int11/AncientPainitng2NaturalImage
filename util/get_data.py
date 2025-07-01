from __future__ import print_function
import os
import tarfile
import requests
from warnings import warn
from zipfile import ZipFile
from bs4 import BeautifulSoup
from os.path import abspath, isdir, join, basename


class GetData(object):
    """

    Download CycleGAN, Pix2Pix, Ancient Painting Data, or GitHub datasets.

    Args:
        technique : str
            One of: 'cyclegan', 'pix2pix', or any string for ancient painting datasets.
        verbose : bool
            If True, print additional information.

    Examples:
        >>> from util.get_data import GetData
        >>> gd = GetData(technique='cyclegan')
        >>> new_data_path = gd.get(save_path='./datasets')  # options will be displayed.
        
        >>> # For ancient painting datasets
        >>> gd = GetData()
        >>> bird_path = gd.get('ancient_painting_bird', './data')
        >>> all_paths = gd.get_dstn_dataset('./data')
        
        >>> # For GitHub dataset
        >>> dlp_path = gd.get_dlp_gan_dataset('./data')

    """

    def __init__(self, verbose=True):
        url_dict = {
            'ancient_painting_bird': 'https://drive.usercontent.google.com/download?id=1G67wmjjunArntetMnxQq0g93BuDSC8XL&export=download&authuser=0&confirm=t&uuid=e6d89ffc-b1aa-4212-871c-e21ea8b7baf1&at=AN8xHoq8U8_c7h5Z_m_wRthSMTyu:1750930357938',
            'ancient_painting_flower': 'https://drive.usercontent.google.com/download?id=1ARzNtGvi_-9woQYckgg835WFhHYbF3kE&export=download&authuser=0&confirm=t&uuid=a1c790e2-04cb-4b3e-abd8-e86171a5fd4f&at=AN8xHoorj5I0yYpB-Xkj8m136AdH:1750930364822',
            'ancient_painting_landscape': 'https://drive.usercontent.google.com/download?id=1K4ujNMTvE-bxWcN-kV-_jRP9S8zfeThp&export=download&authuser=0&confirm=t&uuid=7e56b8a3-8207-4d39-85c3-6fbcb5b338d3&at=AN8xHoo9OOE8MWotvFGpebMiBy1j:1750930367836',
            'traditional_chinese_landscape_painting': 'https://github.com/alicex2020/Chinese-Landscape-Painting-Dataset/archive/refs/heads/main.zip'
        }
        self.url_dict = url_dict  # Store the dict for ancient painting methods
        self._verbose = verbose

    def _print(self, text):
        if self._verbose:
            print(text)

    def _download_data(self, dataset_url, save_path, filename=None):
        if not isdir(save_path):
            os.makedirs(save_path)

        if filename is None:
            base = basename(dataset_url)
        else:
            base = filename
        temp_save_path = join(save_path, base)

        # Check if it's a Google Drive URL and add appropriate headers
        headers = {}
        if 'drive.usercontent.google.com' in dataset_url:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        elif 'github.com' in dataset_url:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/vnd.github.v3+json'
            }

        with open(temp_save_path, "wb") as f:
            r = requests.get(dataset_url, headers=headers, stream=True)
            r.raise_for_status()  # Raise an exception for bad status codes
            
            if 'drive.usercontent.google.com' in dataset_url or 'github.com' in dataset_url:
                # For Google Drive and GitHub, write in chunks
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            else:
                f.write(r.content)

        if base.endswith('.tar.gz'):
            obj = tarfile.open(temp_save_path)
        elif base.endswith('.zip'):
            obj = ZipFile(temp_save_path, 'r')
        else:
            # For Google Drive downloads, try zip first, then tar.gz
            try:
                obj = ZipFile(temp_save_path, 'r')
            except:
                try:
                    obj = tarfile.open(temp_save_path)
                except:
                    raise ValueError("Unknown File Type: {0}.".format(base))

        self._print("Unpacking Data...")
        obj.extractall(save_path)
        obj.close()
        os.remove(temp_save_path)

    def get(self, dataset_key, save_path, dataset_name=None):
        """
        Download a specific dataset by URL key.

        Args:
            dataset_key : str
                Key from url_dict to download (e.g., 'ancient_painting_bird', 'ancient_painting_flower', etc.)
            save_path : str
                A directory to save the data to.
            dataset_name : str, optional
                Custom name for the dataset folder. If None, will derive from dataset_key.

        Returns:
            save_path_full : str
                The absolute path to the downloaded data.
        """
        if dataset_key not in self.url_dict:
            available_keys = list(self.url_dict.keys())
            raise ValueError(f"Unknown dataset key: {dataset_key}. Available keys: {available_keys}")
        
        # If no custom name provided, derive from key
        if dataset_name is None:
            if dataset_key.startswith('ancient_painting_'):
                dataset_name = dataset_key.replace('ancient_painting_', '')
            else:
                dataset_name = dataset_key
        
        save_path_full = join(save_path, dataset_name)
        
        if isdir(save_path_full):
            warn("\n'{0}' already exists. Voiding Download.".format(save_path_full))
        else:
            self._print(f'Downloading {dataset_key} Data...')
            url = self.url_dict[dataset_key]
            self._download_data(url, save_path=save_path, filename=f'{dataset_name}_dataset.zip')

        return abspath(save_path_full)

    def get_dlp_gan_dataset(self, save_path):
        """
        Download the Chinese Landscape Painting Dataset from GitHub.
        
        Args:
            save_path : str
                A directory to save the data to.
                
        Returns:
            save_path_full : str
                The absolute path to the downloaded data.
        """
        if isdir(save_path):
            warn("\n'{0}' already exists. Voiding Download.".format(save_path))
        else:
            self._print('Downloading DLP GAN Dataset from GitHub...')
            url = self.url_dict['traditional_chinese_landscape_painting']
            
            # Create temporary download directory
            temp_download_path = join(os.path.dirname(save_path), 'temp_dlp_download')
            self._download_data(url, save_path=temp_download_path, filename='dlp_gan_dataset.zip')
            
            # GitHub downloads extract to a folder with "-main" suffix
            extracted_folder = join(temp_download_path, 'Chinese-Landscape-Painting-Dataset-main')

            import shutil
            from zipfile import ZipFile
            
            # Path to All-Paintings folder
            all_paintings_path = join(extracted_folder, 'All-Paintings')
            

            # Create target directory if it doesn't exist
            if not isdir(save_path):
                os.makedirs(save_path)
            
            # Move all folders from All-Paintings to save_path
            for item in os.listdir(all_paintings_path):
                src_path = join(all_paintings_path, item)
                dst_path = join(save_path, item)
                
                if isdir(src_path):
                    # If destination exists, remove it first
                    if isdir(dst_path):
                        shutil.rmtree(dst_path)
                    # Move the folder
                    shutil.move(src_path, dst_path)
                    self._print(f'Moved folder: {item} to {save_path}')
        
            self._print(f'DLP GAN Dataset organized in: {save_path}')

            # Clean up temporary download folder
            shutil.rmtree(temp_download_path)
            self._print('Cleaned up temporary files')

            # Extract all zip files in save_path using glob
            import glob
            zip_files = glob.glob(join(save_path, "**/*.zip"), recursive=True)
            for zip_file in zip_files:
                zip_dir = os.path.dirname(zip_file)
                with ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(zip_dir)
                
                # Move jpg files to parent folder and clean up
                for item in os.listdir(zip_dir):
                    item_path = join(zip_dir, item)
                    if isdir(item_path) and item != '__MACOSX':
                        # Find jpg files in this folder
                        jpg_files = glob.glob(join(item_path, "*.jpg"))
                        for jpg_file in jpg_files:
                            shutil.move(jpg_file, join(zip_dir, basename(jpg_file)))
                        # Remove the folder after moving jpgs
                        shutil.rmtree(item_path)
                    elif isdir(item_path) and item == '__MACOSX':
                        # Remove __MACOSX folder
                        shutil.rmtree(item_path)
                
                # Remove the zip file after extraction
                os.remove(zip_file)


            return abspath(save_path)


    def get_dstn_dataset(self, save_path):
        """
        Download all ancient painting datasets.

        Args:
            save_path : str
                A directory to save the data to.

        Returns:
            dataset_paths : dict
                Dictionary containing the absolute paths to all downloaded datasets.
        """
        self._print('Downloading All Ancient Painting Datasets...')

        dataset_paths = {
            'bird': self.get('ancient_painting_bird', save_path),
            'flower': self.get('ancient_painting_flower', save_path),
            'landscape': self.get('ancient_painting_landscape', save_path)
        }

        # Clean up __MACOSX folders if they exist (created by macOS when zipping)
        self._print('Cleaning up __MACOSX folders...')
        import shutil
        macosx_path = join(save_path, '__MACOSX')
        if isdir(macosx_path):
            shutil.rmtree(macosx_path)
            self._print("Removed __MACOSX metadata folder")

        self._print('All Ancient Painting Datasets Downloaded Successfully!')
        return dataset_paths


if __name__ == "__main__":
    print("=== Ancient Painting Dataset Downloader ===\n")
    downloader = GetData(verbose=True)

    print("DLP GAN Dataset Download...")
    try:
        dlp_path = downloader.get_dlp_gan_dataset('./datasets/dlp_gan_dataset')
        print(f"✓ DLP GAN dataset downloaded to: {dlp_path}")
    except Exception as e:
        print(f"✗ Error downloading DLP GAN dataset: {e}")

    print("\n2. Downloading Google Drive Datasets...")
    try:
        all_paths = downloader.get_dstn_dataset('./datasets/dstn_dataset')
        print("✓ All Google Drive datasets downloaded successfully!")
        print("Dataset paths:")
        for dataset_name, path in all_paths.items():
            print(f"  - {dataset_name}: {path}")
    except Exception as e:
        print(f"✗ Error downloading Google Drive datasets: {e}")
