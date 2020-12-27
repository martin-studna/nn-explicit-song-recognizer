<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/martin-studna/nn-explicit-song-recognizer">
    <img src="https://www.freepnglogos.com/uploads/spotify-logo-png/file-spotify-logo-png-4.png" alt="Logo" width="100" height="100">
  </a>

  <h3 align="center">Explicit Song Recognizer</h3>

  <p align="center">
    Neural networks recognizing explicit songs in Spotify dataset
    <br />
    <br />
    <br />
    <a href="https://github.com/martin-studna/nn-explicit-song-recognizer/issues">Report Bug</a>
    ·
    <a href="https://github.com/martin-studna/nn-explicit-song-recognizer/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This project was developed with Jan Babušík as a solution to the semester project from Neural networks master course. We decided to take Spotify dataset and create neural networks in Python, which can learn which songs from the dataset are explicit or not.

### Built With

- [Python](https://www.python.org/)
- [Numpy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Sklearn](https://scikit-learn.org/)

<!-- GETTING STARTED -->

## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/martin-studna/nn-explicit-song-recognizer.git
   ```
2. Install dependencies
   ```sh
   pip3 install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->

## Usage

Example:

```sh
   python3 recognizer.py
```

Optional paramters:

```sh
  --epochs          Number of epochs for training
  --lr              Learning rate
  --hidden_size     Size of the hidden layer
  --seed            Random seed
  --batch_size      Batch size
  --test_size       Size of the test set
  --plot_conf       Plot confusion matrix
```

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->

## Contact

- Martin Studna - martin.studna2@gmail.com

- Jan Babušík - email

<!-- ACKNOWLEDGEMENTS -->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/martin-studna/nn-explicit-song-recognizer.svg?style=for-the-badge
[contributors-url]: https://github.com/martin-studna/nn-explicit-song-recognizer/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/martin-studna/nn-explicit-song-recognizer.svg?style=for-the-badge
[forks-url]: https://github.com/martin-studna/nn-explicit-song-recognizer/network/members
[stars-shield]: https://img.shields.io/github/stars/martin-studna/nn-explicit-song-recognizer.svg?style=for-the-badge
[stars-url]: https://github.com/martin-studna/nn-explicit-song-recognizer/stargazers
[issues-shield]: https://img.shields.io/github/issues/martin-studna/nn-explicit-song-recognizer.svg?style=for-the-badge
[issues-url]: https://github.com/martin-studna/nn-explicit-song-recognizer/issues
[license-shield]: https://img.shields.io/github/license/martin-studna/nn-explicit-song-recognizer.svg?style=for-the-badge
[license-url]: https://github.com/martin-studna/nn-explicit-song-recognizer/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/martin-studna
