"""課程包清單：安裝器與測試的唯一來源（ADR 0029）。

每個課程包列出指令名稱、側欄分段名稱、擁有的模板路徑與額外套件。
路徑相對於模板的 ``dataset_streamlit_shell/``；資料夾代表其下所有檔案。
不在任何課程包內的模板檔即 AI Lab 框架。
現階段允許一個路徑暫時出現在多個課程包（之後各帶副本，見 #12–#16）。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CoursePack:
    name: str
    sidebar_section: str
    paths: tuple[str, ...]
    dependencies: tuple[str, ...] = ()


DATA_SCIENCE = CoursePack(
    name="data-science",
    sidebar_section="AI 協作資料整理",
    paths=(
        "pages/3_Field_Quality.py",
        "pages/17_Data_Transform.py",
        "pages/4_Duplicates.py",
        "pages/5_Numeric_Diagnostics.py",
        "pages/6_Outliers.py",
        "pages/7_Categorical.py",
        "pages/8_Encoding.py",
        "pages/9_Correlation.py",
        "pages/13_Feature_Scaling.py",
        "pages/2_Charts.py",
        "pages/20_Data_Split.py",
        "ui/workflow_ui.py",
        "ui/data_science.py",
        "ml/integration.py",
        "ml/split.py",
        "assets/correlation_formula.png",
        "assets/encoding_onehot_vs_label.png",
        "built-in-data/integration",
    ),
    dependencies=("matplotlib", "scikit-learn"),
)

SUPERVISED = CoursePack(
    name="supervised",
    sidebar_section="機器學習 · 監督式",
    paths=(
        "pages/14_Simple_Linear_Regression.py",
        "pages/16_Logistic_Regression.py",
        "pages/18_Linear_SVM.py",
        "pages/19_KNN_Classification.py",
        "pages/19_Decision_Tree_Concepts.py",
        "ui/lr_ui.py",
        "ui/lr_slot_state.py",
        "ui/simple_regression_quiz.py",
        "ui/multiple_regression_quiz.py",
        "ui/teaching_flow.py",
        "ui/lr_blocks_prototype.py",
        "ui/lr_blocks_prototype.html",
        "ui/logistic_regression_ui.py",
        "ui/logistic_slot_state.py",
        "ui/logistic_quiz.py",
        "ui/svm_ui.py",
        "ui/svm_quiz.py",
        "ui/knn_ui.py",
        "ui/knn_quiz.py",
        "ui/tree_ui.py",
        "ui/tree_ensemble_quiz.py",
        "ml/regression.py",
        "ml/classification.py",
        "ml/svm.py",
        "ml/knn.py",
        "ml/decision_tree.py",
        "ml/random_forest.py",
        "ml/xgboost_model.py",
        "built-in-data/regression",
        "built-in-data/classification/cat_toy_10.csv",
        "built-in-data/classification/heart_disease.csv",
        "built-in-data/classification/knn_blobs_80.csv",
        "built-in-data/classification/knn_scale_trap_80.csv",
        "built-in-data/classification/microchip_test.csv",
        "built-in-data/classification/svm_blobs_80.csv",
        "built-in-data/classification/svm_soft_margin_80.csv",
        "built-in-data/classification/university_admission.csv",
    ),
    dependencies=("numpy", "matplotlib", "plotly", "scikit-learn", "xgboost"),
)

UNSUPERVISED = CoursePack(
    name="unsupervised",
    sidebar_section="機器學習 · 非監督式",
    paths=(
        "pages/11_Wards_Method.py",
        "pages/12_KMeans.py",
        "ui/clustering_ui.py",
        "ml/clustering.py",
        "built-in-data/clustering",
    ),
    dependencies=("matplotlib", "scipy", "scikit-learn"),
)

DEEP_LEARNING = CoursePack(
    name="deep-learning",
    sidebar_section="深度學習",
    paths=(
        "pages/21_Neural_Network.py",
        "ui/nn_ui.py",
        "ui/nn_form_state.py",
        "ml/coffee_nn.py",
        "ml/tf_runtime.py",
        "built-in-data/classification/nn_binary_400.csv",
    ),
    dependencies=("numpy", "matplotlib", "tensorflow-cpu"),
)

COMPUTER_VISION = CoursePack(
    name="computer-vision",
    sidebar_section="電腦視覺",
    paths=(
        "pages/27_CNN_Introduction.py",
        "pages/22_Image_Classification.py",
        "pages/23_Object_Detection.py",
        "pages/24_Semantic_Segmentation.py",
        "pages/25_Instance_Segmentation.py",
        "pages/26_Promptable_Segmentation.py",
        "ui/cnn_ui.py",
        "ui/cnn_quiz.py",
        "ui/computer_vision_pack.py",
        "ui/cv_layout.py",
        "ui/classification_ui.py",
        "ui/detection_ui.py",
        "ui/semantic_segmentation_ui.py",
        "ui/instance_segmentation_ui.py",
        "ui/promptable_segmentation_ui.py",
        "ml/cnn_intro.py",
        "ml/cnn_pytorch.py",
        "cv",
        "built-in-data/computer-vision",
    ),
    dependencies=(
        "numpy",
        "matplotlib",
        "scikit-learn",
        "tensorflow-cpu",
        "pillow",
        "opencv-python-headless",
        "ultralytics>=8.3.237",
        "clip @ git+https://github.com/ultralytics/CLIP.git",
        "torch",
        "torchvision",
        "gdown",
    ),
)

PCA = CoursePack(
    name="pca",
    sidebar_section="降維分析",
    paths=(
        "pages/10_PCA.py",
        "ui/workflow_ui.py",
        "ui/data_science.py",
        "ml/integration.py",
        "ml/split.py",
    ),
    dependencies=("numpy", "matplotlib", "scikit-learn"),
)

STARTUP_WORKSHOP = CoursePack(
    name="startup-workshop",
    sidebar_section="AI新創工作坊",
    paths=(
        "pages/30_Startup_Challenge.py",
        "ui/startup_challenge_page.py",
        "ui/startup_challenge_context.py",
        "ui/startup_challenge_ui.py",
        "ui/startup_challenge_empty_shell.py",
        "ml/regression.py",
        "ml/classification.py",
        "workspace/challenge",
    ),
    dependencies=("numpy",),
)

COURSE_PACKS: tuple[CoursePack, ...] = (
    DATA_SCIENCE,
    SUPERVISED,
    UNSUPERVISED,
    DEEP_LEARNING,
    COMPUTER_VISION,
    PCA,
    STARTUP_WORKSHOP,
)


ALL_PACKS = "all"


class UnknownCoursePackError(ValueError):
    """指令名稱不在課程包清單內。"""


def resolve_course_packs(names: Iterable[str]) -> tuple[CoursePack, ...]:
    """把指令名稱（或 ``all``）轉成課程包，依清單順序、去重；有未知名稱就報錯並列出可用名稱。"""
    by_name = {pack.name: pack for pack in COURSE_PACKS}
    wanted = list(names)
    unknown = [name for name in wanted if name != ALL_PACKS and name not in by_name]
    if unknown:
        available = ", ".join([*by_name, ALL_PACKS])
        raise UnknownCoursePackError(
            f"Unknown course pack: {', '.join(unknown)}. Available: {available}"
        )
    if ALL_PACKS in wanted:
        return COURSE_PACKS
    return tuple(pack for pack in COURSE_PACKS if pack.name in wanted)


def _template_files(shell_root: Path) -> list[str]:
    return sorted(
        path.relative_to(shell_root).as_posix()
        for path in shell_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )


def _covers(owned: str, rel: str) -> bool:
    return rel == owned or rel.startswith(owned.rstrip("/") + "/")


def _owned(rel: str, packs: Iterable[CoursePack]) -> bool:
    return any(_covers(owned, rel) for pack in packs for owned in pack.paths)


def pack_files(shell_root: Path, pack: CoursePack) -> list[str]:
    """課程包擁有的模板檔（資料夾展開成其下所有檔案），相對於 shell_root。"""
    return [rel for rel in _template_files(shell_root) if _owned(rel, (pack,))]


def framework_files(shell_root: Path) -> list[str]:
    """不屬於任何課程包的模板檔，即 AI Lab 框架。"""
    return [rel for rel in _template_files(shell_root) if not _owned(rel, COURSE_PACKS)]
