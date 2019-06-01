from setuptools import setup, find_packages

setup(name='dnn-classifier',
      version='0.1',
      description='Example implementation of DNN using logistic regression',
      long_description='Lalka',
      classifiers=[
          'Development Status :: 1 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7',
          'Topic :: Image Processing :: DNN',
      ],
      keywords='dnn example',
      url='https://github.com/okeer/dnn-classifier',
      author='Sergey Orlov',
      author_email='serhio.orlovsky@gmail.com',
      license='MIT',
      packages=find_packages(include=['dnn*']),
      install_requires=[
          'h5py', 'numpy', 'pillow'
      ],
      include_package_data=True,
      zip_safe=False)
