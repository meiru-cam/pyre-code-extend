Design the offline pipeline that turns a newly acquired, unlabeled image corpus into two deliverables for a computer-vision team: (1) a set of images that are novel relative to the data the team's current vision model was already trained on, to be added to the next training round, and (2) for a short list of target objects, the images in the corpus that contain them.

The team already has a trained vision encoder — a frozen model that maps an image to a $d = 768$-dimensional embedding — and the $N_{\text{train}} = 4\times10^8$ labeled images it was trained on, each with its embedding already computed and stored from that training run. A data-collection partner delivers a new corpus scraped from the public web: $N_{\text{raw}} = 8\times10^9$ images, unlabeled, averaging 100 KB each as crawled (JPEG, resized to a web-friendly resolution by the crawler), reachable by URL and already sitting in object storage.

For the target-object task, the vision team supplies a list of 200 target object categories; each comes with either 3-10 example images or a one-paragraph text description, never large amounts of either.

Scale given for this design: 128 GPUs (A100-class) are available, and the compute-heavy stages of the pipeline (filtering, embedding, nearest-neighbor scoring) have up to 5 days of that fleet to work with; the whole project, from receiving the corpus to delivering both outputs, has a 2-week deadline, the rest of which covers human-labeling turnaround and review. Up to 6,000 images total can be sent to human annotators across both deliverables.

Out of scope: training or fine-tuning the frozen vision encoder itself (only lightweight heads or classifiers may be trained on top of its output); the crawler and the object-storage layer that produced the corpus; OCR or format-specific parsing; serving either deliverable as a live, user-facing search product — both are offline batch jobs whose output is a manifest, not a request/response API with a latency target.

Produce:

- A requirements and scale estimate: GPU-time to embed the corpus once, the size of the stored vectors, the memory of the nearest-neighbor indexes involved, and the cost of the de-duplication stage.
- A record schema and the pipeline's stage-by-stage interfaces: what each stage reads, what it writes, and what fields survive to the next stage.
- An architecture diagram, plus a walk-through of one image's path through the novelty side and one target object's path through the retrieval side.
- Deep dives into: how "novel" is defined and measured; how target-object retrieval is built, including how to estimate its precision and recall without ground truth over the whole corpus; and how the pipeline is kept affordable and re-runnable at this scale. For each, compare at least two alternatives, say which you would pick, and state the cost of that choice.
