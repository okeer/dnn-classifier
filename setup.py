from setuptools import setup, find_packages

setup(name='numdl',
      version='0.0.1',
      description='NUmpy Deel Learning library',
      long_description='Just a lowlevel implementation of DNN using logistic regression with numpy under the hood',
      classifiers=[
          'Development Status :: 1 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7',
          'Topic :: Image Processing :: Deep Learning',
      ],
      keywords='adel example',
      url='https://github.com/okeer/numdl',
      author='Sergey Orlov',
      author_email='serhio.orlovsky@gmail.com',
      license='MIT',
      packages=find_packages(exclude=['test*']),
      install_requires=[
          'h5py', 'numpy', 'pillow'
      ],
      include_package_data=True,
      zip_safe=False)
