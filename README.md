# Global Renewables Watch

We present a comprehensive global temporal dataset of commercial solar photovoltaic (PV) farms and onshore wind turbines, derived from high-resolution satellite imagery analyzed quarterly from the fourth quarter of 2017 to the second quarter of 2024. We create this dataset by training deep learning based segmentation models to identify these renewable energy installations from satellite imagery, then deploy them on over 13 trillion pixels covering the world. For each detected feature, we estimate the construction date and the preceding land use type. This dataset offers crucial insights into progress toward sustainable development goals and serves as a valuable resource for policymakers, researchers, and stakeholders aiming to assess and promote effective strategies for renewable energy deployment. Our final spatial dataset includes 375,197 individual wind turbines and 86,410 solar PV installations. We aggregate our predictions to the country level --- estimating total power capacity based on construction date, solar PV area, and number of windmills --- and find a R2 values of 0.96 and 0.93 for solar PV and onshore wind respectively compared to IRENA's most recent 2023 country level capacity estimates.

## Dataset Download

You can download the dataset from the release page, [here](https://github.com/microsoft/global-renewables-watch/releases/tag/v1.0).


## Citation

If you use this work, please consider citing our [paper](https://arxiv.org/abs/2503.14860):
```
@article{robinson2025global,
  title={Global Renewables Watch: A Temporal Dataset of Solar and Wind Energy Derived from Satellite Imagery},
  author={Robinson, Caleb and Ortiz, Anthony and Kim, Allen and Dodhia, Rahul and Zolli, Andrew and Nagaraju, Shivaprakash K and Oakleaf, James and Kiesecker, Joe and Ferres, Juan M Lavista},
  journal={arXiv preprint arXiv:2503.14860},
  year={2025}
}
```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
