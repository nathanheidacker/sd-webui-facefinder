import modules.scripts as scripts
import gradio as gr
import subprocess as sp
import numpy as np

import facefinder
import platform
import sys
import cv2
import os

from modules import script_callbacks, shared
from modules.extras import run_pnginfo
from gradio.events import EventListenerMethod
from pathlib import Path
from functools import partial
from PIL import Image, PngImagePlugin
from tqdm import tqdm

from typing import Literal, Any

folder_symbol = "\U0001f4c2"  # 📂
trashcan_symbol = "\U0001F5D1"  # 🗑️

ROOT = Path(scripts.basedir()).joinpath("FaceFinder").resolve()
PATHS = {"ROOT": ROOT}

facefinder.set_paths(**PATHS)
facefinder.create_dirs()


def set_target(target: str) -> None:
    facefinder.PATHS.__class__.TARGET_IMAGE = Path(target).resolve()


def pretty_print_dict(d: dict, n_tabs: int = 0) -> str:
    result = []
    tabs = " " * 4 * n_tabs
    for k, v in d.items():
        numeric = False
        subtype = ""
        if isinstance(v, np.ndarray):
            try:
                v / 4
                numeric = True
            except TypeError:
                subtype = str(type(v.flatten()[0]))
                subtype = f"[{subtype[8:-2]}]"

        if numeric:
            array_format = f"ndarray {v.shape} | mean: {v.mean():0.3f} | max: {v.max():0.3f} | min: {v.min():0.3f}"
            result.append(f"{tabs}{k}: {array_format}")
        elif isinstance(v, dict):
            result.append(f"{tabs}{k}:\n{pretty_print_dict(v, n_tabs+1)}")
        elif "\n" not in str(v):
            result.append(f"{tabs}{k}: {str(v)[:500]}")
        elif isinstance(v, list):
            result.append(f"{tabs}{k}: list | len: {len(v)}")
        else:
            result.append(f"{tabs}{k}: {str(type(v))[8:-2]}{subtype}")
    return ("\n" * (1 if n_tabs else 2)).join(result)


def empty_dir(path: str) -> None:
    path = Path(path).resolve()
    for thing in os.listdir(path):
        thing = path.joinpath(thing)
        if os.path.isdir(thing):
            empty_dir(str(thing))
        else:
            os.remove(thing)


def set_path(path: str, pathname: str) -> None:
    paths = PATHS | {pathname: Path(path).resolve()}
    facefinder.set_paths(absolute=True, **paths)


def copy_images(indir: str, outdir: str) -> None:
    outdir = Path(outdir).resolve()
    indir = Path(indir).resolve()
    files = os.listdir(indir)
    for file in tqdm(files, desc=f"copying {len(files)} files"):
        if file.endswith((".png", ".jpg", ".jpeg")):
            image_path = indir.joinpath(file)
            image = Image.open(image_path)
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
            image.save(outdir.joinpath(file), pnginfo=metadata)


def open_folder(f):
    if not os.path.exists(f):
        print(f'Folder "{f}" does not exist')
        return
    elif not os.path.isdir(f):
        print(
            f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""",
            file=sys.stderr,
        )
        return

    if not shared.cmd_opts.hide_ui_dir_config:
        path = os.path.normpath(f)
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            sp.Popen(["open", path])
        elif "microsoft-standard-WSL2" in platform.uname().release:
            sp.Popen(["wsl-open", path])
        else:
            sp.Popen(["xdg-open", path])


def empty_metrics() -> dict:
    return {
        "paths": {"target": None, "embedding": [], "input": []},
        "representations": {},
        "distances": {
            "embedding_target": [],
            "embedding_embedding": [],
            "candidate_target": [],
            "candidate_embedding": [],
            "pairs": [],
        },
        "indices": {"match": [], "cumulative": [], "target": []},
        "metrics": {
            "embedding_accuracy": 0,
            "embedding_coherence": 0,
            "candidate_weighted_distances": [],
            "embedding_weighted_distances": [],
            "candidate_scores": [],
            "embedding_scores": [],
        },
        "metadata": {"model": [], "target_distance_bias": 0},
    }


def get_n_images() -> tuple[int, int, int]:
    n_embedding = len(os.listdir(facefinder.PATHS.EMBEDDING_IMAGES))
    n_unprocessed = len(os.listdir(facefinder.PATHS.UNPROCESSED_IMAGES))
    n_processed = len(os.listdir(facefinder.PATHS.PROCESSED_IMAGES))
    return {
        "n_embedding": n_embedding,
        "n_unprocessed": n_unprocessed,
        "n_processed": n_processed,
    }


class State:
    def __init__(self) -> None:
        self._metrics = empty_metrics()

        self.embedding_sort_idx = np.array([])
        self.embedding_selected_idx = 0
        self.embedding_selected_path = ""

        self.candidate_sort_idx = np.array([])
        self.candidate_selected_idx = 0
        self.candidate_selected_path = ""

    def __str__(self) -> str:
        n_embedding, n_unprocessed, n_processed = get_n_images().values()
        return pretty_print_dict(self.metrics)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, str(item))

    def __setitem__(self, name: str, item: Any) -> None:
        setattr(self, name, item)

    @property
    def metrics(self) -> dict:
        return self._metrics

    def analyze(self, target_distance_bias: int, models: list[str]) -> None:
        if not models:
            models = ["Facenet512"]
        self._metrics = facefinder.analyze_faces(
            models, target_distance_bias=target_distance_bias
        )

    def get_gallery_paths(
        self,
        gallery: Literal["embedding", "candidate"],
        sort_by: Literal["score", "target", "embedding", "weighted"],
    ):
        idx = self.metrics["indices"][f"{gallery}_{sort_by}_idx"]
        self[f"{gallery}_sort_idx"] = idx
        paths = self.metrics["paths"][gallery][idx]
        return [str(path) for path in paths[:2000]]

    def get_image_stats(
        self, event: gr.SelectData, gallery: Literal["embedding", "candidate"]
    ) -> str:
        idx = self[f"{gallery}_sort_idx"][event.index]
        path = self.metrics["paths"][gallery][idx]
        _, gen_info, _ = run_pnginfo(Image.open(path))
        self[f"{gallery}_selected_idx"] = idx
        self[f"{gallery}_selected_path"] = path

        scores = self.metrics["metrics"][f"{gallery}_scores"]
        score = scores[idx]
        score_percentile = (score > scores).mean()

        target_distances = self.metrics["distances"][f"{gallery}_target"]
        target_distance = float(target_distances[idx][0])
        target_distance_percentile = (target_distance > target_distances).mean()

        embedding_distances = self.metrics["distances"][f"{gallery}_embedding"]
        embedding_distance = float(embedding_distances[idx][0])
        embedding_distance_percentile = (
            embedding_distance > embedding_distances
        ).mean()

        weighted_distances = self.metrics["metrics"][f"{gallery}_weighted_distances"]
        weighted_distance = weighted_distances[idx]
        weighted_distance_percentile = (
            weighted_distance > np.array(weighted_distances)
        ).mean()

        return (
            f"TARGET DISTANCE:    {target_distance:.4f} | Top {target_distance_percentile*100:.2f}%\n"
            f"EMBEDDING DISTANCE: {embedding_distance:.4f} | Top {embedding_distance_percentile*100:.2f}%\n"
            f"WEIGHTED DISTANCE:  {weighted_distance:.4} | Top {weighted_distance_percentile*100:.2f}%\n\n"
            f"SCORE:              {score:.4f} | Top {score_percentile*100:.2f}%\n\n"
            f"METADATA:\n    filename: {path.name}\n    fullpath: {str(path)}"
            f"\n\n{gen_info}"
        )


state = State()
output = gr.TextArea(
    state.__str__,
    label="Analysis",
    interactive=False,
    every=None,
    elem_classes="raw_output",
)

embedding_sort_by = gr.Dropdown(
    ["score", "target distance", "embedding distance", "weighted distance"],
    value="score",
    label="Sort by",
    interactive=True,
)
embedding_gallery = gr.Gallery(label="Embedding Images").style(grid=4, container=True)
embedding_target_viewer = gr.Image(show_label=False, interactive=False)

candidate_sort_by = gr.Dropdown(
    ["score", "target distance", "embedding distance", "weighted distance"],
    value="score",
    label="Sort by",
    interactive=True,
)
candidate_gallery = gr.Gallery(label="Candidate Images").style(grid=4, container=True)
candidate_target_viewer = gr.Image(show_label=False, interactive=False)


vault_gallery = gr.Gallery(label="Vaulted Images").style(grid=4, container=True)
vault_target_viewer = gr.Image(show_label=False, interactive=False)


def analyze(target_distance_bias: int, models: list[str], state: State) -> dict:
    if not models:
        models = ["Facenet512"]
    state.metrics = facefinder.analyze_faces(
        models, target_distance_bias=target_distance_bias
    )


def refresh(component):
    return (
        component.then(fn=state.__str__, outputs=[output])
        .then(
            fn=lambda sb: state.get_gallery_paths("embedding", sb.split(" ")[0]),
            inputs=[embedding_sort_by],
            outputs=[embedding_gallery],
        )
        .then(
            fn=lambda sb: state.get_gallery_paths("candidate", sb.split(" ")[0]),
            inputs=[candidate_sort_by],
            outputs=[candidate_gallery],
        )
    )


def refresh_vault(component):
    return component.then(
        fn=lambda: [
            str(facefinder.PATHS.PROCESSED_IMAGES.joinpath(img))
            for img in os.listdir(facefinder.PATHS.PROCESSED_IMAGES)
        ],
        outputs=[vault_gallery],
    )


def on_ui_tabs():
    with gr.Blocks(
        analytics_enabled=False,
    ) as ui_component:
        with gr.Tab(label="Analyze"):
            with gr.Row():
                # Input column (left side)
                with gr.Column():
                    # Target image
                    target_image = gr.Image(
                        label="Target Image",
                        elem_id="target_image_input",
                        type="filepath",
                    )

                    refresh_vault(
                        target_image.upload(
                            fn=lambda target: set_target(target), inputs=[target_image]
                        ).then(
                            fn=lambda: [
                                facefinder.PATHS.TARGET_IMAGE for _ in range(3)
                            ],
                            outputs=[
                                embedding_target_viewer,
                                candidate_target_viewer,
                                vault_target_viewer,
                            ],
                        )
                    )

                    # Embedding images control panel
                    with gr.Row():
                        embedding_images_dir = gr.Textbox(
                            label="Copy embedding images",
                            elem_id="embedding_images_dir",
                            placeholder="Path to directory with embedding images to be copied",
                        ).style(show_copy_button=True)

                        embedding_copy_button = gr.Button(
                            "COPY",
                            elem_classes=["tool"],
                        )

                        refresh(
                            embedding_copy_button.click(
                                fn=lambda images: copy_images(
                                    images, facefinder.PATHS.EMBEDDING_IMAGES
                                ),
                                inputs=[embedding_images_dir],
                            )
                        )

                        embedding_folder_button = gr.Button(
                            folder_symbol,
                            visible=not shared.cmd_opts.hide_ui_dir_config,
                            elem_classes=["tool"],
                        )

                        refresh(
                            embedding_folder_button.click(
                                fn=lambda: open_folder(
                                    facefinder.PATHS.EMBEDDING_IMAGES
                                ),
                            )
                        )

                        embedding_empty_button = gr.Button(
                            trashcan_symbol, elem_classes=["tool"]
                        )

                        refresh(
                            embedding_empty_button.click(
                                fn=lambda: empty_dir(facefinder.PATHS.EMBEDDING_IMAGES),
                            )
                        )

                    # Unprocessed images control panel
                    with gr.Row():
                        unprocessed_images_dir = gr.Textbox(
                            label="Copy new face images",
                            elem_id="unprocessed_images_dir",
                            placeholder="Path to directory with face images to be copied",
                        ).style(show_copy_button=True)

                        unprocessed_copy_button = gr.Button(
                            "COPY",
                            elem_classes=["tool"],
                        )

                        refresh(
                            unprocessed_copy_button.click(
                                fn=lambda images: copy_images(
                                    images, facefinder.PATHS.UNPROCESSED_IMAGES
                                ),
                                inputs=[unprocessed_images_dir],
                            )
                        )

                        unprocessed_folder_button = gr.Button(
                            folder_symbol,
                            visible=not shared.cmd_opts.hide_ui_dir_config,
                            elem_classes=["tool"],
                        ).style(full_width=False)

                        refresh(
                            unprocessed_folder_button.click(
                                fn=lambda: open_folder(
                                    facefinder.PATHS.UNPROCESSED_IMAGES
                                ),
                            )
                        )

                        unprocessed_empty_button = gr.Button(
                            trashcan_symbol, elem_classes=["tool"]
                        )

                        refresh(
                            unprocessed_empty_button.click(
                                fn=lambda: empty_dir(
                                    facefinder.PATHS.UNPROCESSED_IMAGES
                                ),
                            )
                        )

                    # Analyzer control panel
                    with gr.Row():
                        # Selection of weight relevance
                        target_distance_bias = gr.Slider(
                            minimum=1,
                            maximum=2,
                            step=0.01,
                            value=1.4,
                            interactive=True,
                            label="Weight Relevance",
                        )

                        # Selection of models

                        with gr.Accordion(label="Model Selection", open=False):
                            model_checklist = gr.CheckboxGroup(
                                facefinder.metadata.MODELS,
                                show_label=False,
                                value=["Facenet512"],
                                info="Select at least one model",
                                interactive=True,
                            )

                    # Analyze / Clearing cache (intermediate files like extracted faces / reprs)
                    with gr.Row():
                        analyze_button = gr.Button(
                            "Analyze", elem_id="analyze_button", variant="primary"
                        )

                        refresh(
                            analyze_button.click(
                                fn=state.analyze,
                                inputs=[target_distance_bias, model_checklist],
                            )
                        )

                        def empty_cache():
                            empty_dir(facefinder.PATHS.REPRESENTATIONS_DIR)
                            empty_dir(facefinder.PATHS.EXTRACTED_FACES_DIR)

                        gr.Button("Clear Cached Data").click(empty_cache)

                # Output / Display column (right side)
                with gr.Column(variant="panel"):
                    with gr.Row():
                        output.render()

                    refresh_button = gr.Button("Refresh").click(
                        fn=state.__str__, outputs=[output]
                    )

        with gr.Tab(label="Browse Embedding"):
            with gr.Row():
                with gr.Column():
                    embedding_gallery.render()

                    with gr.Row():
                        embedding_sort_by.render()

                        embedding_sort_by.select(
                            fn=lambda sb: state.get_gallery_paths(
                                "embedding", sb.split(" ")[0]
                            ),
                            inputs=[embedding_sort_by],
                            outputs=[embedding_gallery],
                        )

                        add_to_vault = gr.Button("Add to Vault")

                        refresh_vault(
                            add_to_vault.click(
                                fn=lambda: cv2.imwrite(
                                    str(
                                        facefinder.PATHS.PROCESSED_IMAGES.joinpath(
                                            state.embedding_selected_path.name
                                        )
                                    ),
                                    cv2.imread(str(state.embedding_selected_path)),
                                )
                            )
                        )

                with gr.Column():
                    with gr.Accordion(label="Target Image", open=False):
                        embedding_target_viewer.render()

                    embedding_image_info = gr.TextArea(
                        label="Selected Image Information",
                        interactive=False,
                        elem_classes="raw_output",
                    )

                    embedding_gallery.select(
                        fn=partial(state.get_image_stats, gallery="embedding"),
                        outputs=[embedding_image_info],
                    )

        with gr.Tab(label="Browse Images"):
            with gr.Row():
                with gr.Column():
                    candidate_gallery.render()

                    with gr.Row():
                        candidate_sort_by.render()

                        candidate_sort_by.select(
                            fn=lambda sb: state.get_gallery_paths(
                                "candidate", sb.split(" ")[0]
                            ),
                            inputs=[candidate_sort_by],
                            outputs=[candidate_gallery],
                        )

                        add_to_vault = gr.Button("Add to Vault")

                        refresh_vault(
                            add_to_vault.click(
                                fn=lambda: cv2.imwrite(
                                    str(
                                        facefinder.PATHS.PROCESSED_IMAGES.joinpath(
                                            state.candidate_selected_path.name
                                        )
                                    ),
                                    cv2.imread(str(state.candidate_selected_path)),
                                )
                            )
                        )

                with gr.Column():
                    with gr.Accordion(label="Target Image", open=False):
                        candidate_target_viewer.render()

                    candidate_image_info = gr.TextArea(
                        label="Selected Image Information",
                        interactive=False,
                        elem_classes="raw_output",
                    )

                    candidate_gallery.select(
                        fn=partial(state.get_image_stats, gallery="candidate"),
                        outputs=[candidate_image_info],
                    )

        with gr.Tab(label="Vault"):
            with gr.Row():
                with gr.Column():
                    vault_gallery.render()

                    with gr.Row():
                        vault_open_button = gr.Button(
                            "Open Vaulted Images Dir",
                        )

                        refresh(
                            refresh_vault(
                                vault_open_button.click(
                                    fn=lambda: open_folder(
                                        facefinder.PATHS.PROCESSED_IMAGES
                                    )
                                )
                            )
                        )

                        vault_empty_button = gr.Button(
                            "Clear Vaulted Images Dir",
                        )

                        refresh(
                            refresh_vault(
                                vault_empty_button.click(
                                    fn=lambda: empty_dir(
                                        facefinder.PATHS.PROCESSED_IMAGES
                                    )
                                )
                            )
                        )

                    def overwrite():
                        empty_dir(facefinder.PATHS.EMBEDDING_IMAGES)
                        copy_images(
                            facefinder.PATHS.PROCESSED_IMAGES,
                            facefinder.PATHS.EMBEDDING_IMAGES,
                        )

                    gr.Button(
                        "Overwrite Embedding with Vault", variant="primary"
                    ).click(overwrite).then(
                        fn=lambda sb: state.get_gallery_paths(
                            "embedding", sb.split(" ")[0]
                        ),
                        inputs=[embedding_sort_by],
                        outputs=[embedding_gallery],
                    )

                with gr.Column():
                    with gr.Accordion(label="Target Image", open=False):
                        vault_target_viewer.render()

        return [(ui_component, "FaceFinder", "facefinder_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)
