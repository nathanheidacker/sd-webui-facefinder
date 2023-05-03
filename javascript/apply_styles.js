
function apply_style() {
    let raw_outputs = gradioApp().querySelectorAll(".raw_output>label>textarea")
    for (let output of raw_outputs) {
        output.style.fontFamily = "monospace"
        output.style.whiteSpace = "pre"
        output.style.overflowX = "scroll"
    }
}

onUiLoaded(apply_style)