# Third-party notices

Reorganizing and renaming the project does not remove upstream authorship or license obligations.

| Component | Source | Scope and preserved terms |
|---|---|---|
| MCI architecture, attentive neck, decoder, original data conventions | [Chen-Yang-Liu/Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent), commit `68cbaa7f388b36e4fc10872f7a2911482d26ae5b` | Upstream MIT notice: [license copy](third_party/licenses/change-agent.txt). The upstream README also says academic use; retain this provenance rather than assuming it resolves other component terms. |
| SegFormer / Mix Vision Transformer implementation and backbone | [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) via Change-Agent | NVIDIA copyright and source license, including non-commercial use: [full license](third_party/licenses/segformer-nvidia.txt). Applies to the work and derivatives according to its terms. |
| BLEU, ROUGE-L, CIDEr implementations | COCO-caption implementations distributed by Change-Agent | Author notices remain in `src/urban_change/evaluation/_vendor/`; [MIT notice](third_party/licenses/coco-caption.txt). |
| Original LEVIR-MCI image pairs in the README | Original archive hosted in the linked dataset repository, revision `3844bd00748c4d79af90e7a002f919b0dd20bb13` | Original PNG bytes are unchanged. Generated masks and captions are this checkpoint’s outputs, clearly distinguished from ground truth. [File provenance](docs/assets/examples/provenance.json). |
| Optional METEOR | Michael Denkowski / official METEOR 1.5 distribution | Not bundled in the package. Obtain and review its distribution terms separately before use or redistribution. |
| Original LEVIR-MCI imagery and annotations | LEVIR-MCI / Change-Agent authors | Dataset and imagery rights are distinct from a source-code license. Confirm applicable redistribution/use terms. |

The project’s temporal reversal pipeline, training adaptations, packaging, services, tests, and documentation build on these components. New documentation is written for this project; foundational papers are cited rather than copied into the repository as PDFs.

The existing model Hub metadata declares Apache-2.0 and the dataset Hub metadata declares MIT. Those declarations were observed in the linked repositories; they do not establish compatibility with all upstream code, pretrained weights, or imagery rights. Resolve the scope before advertising an unrestricted release. This reorganization does not invent a new license for external artifacts.
