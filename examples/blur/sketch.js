let blurRenderer
let radiusSlider
let sigmaSlider
let radiusValue
let sigmaValue
let statusLine

function setup() {
  createCanvas(400, 400, WEBGL)

  radiusSlider = createSlider(0, 64, 10, 1)
  radiusValue = makeSliderRow('radius', radiusSlider)

  // sigma starts at 0 on purpose: 0 (and any non-positive/NaN value) is an
  // invalid sigma, so the renderer falls back to a radius-derived default
  // instead of blowing up weight normalization.
  sigmaSlider = createSlider(0, 20, 0, 0.1)
  sigmaValue = makeSliderRow('sigma (0 = auto fallback)', sigmaSlider)

  statusLine = createDiv('')
  statusLine.class('status')

  blurRenderer = createGaussianBlurRenderer({
    radius: radiusSlider.value(),
    sigma: sigmaSlider.value(),
  })
}

function makeSliderRow(label, slider) {
  const row = createDiv()
  row.class('slider-row')
  const labelSpan = createSpan(label)
  labelSpan.parent(row)
  slider.parent(row)
  const value = createSpan('')
  value.parent(row)
  return value
}

function draw() {
  // Push the slider values through every frame so the blur responds live.
  // Radius 0 takes the passthrough path and never touches the two intermediate
  // render targets.
  blurRenderer.setRadius(radiusSlider.value())
  blurRenderer.setSigma(sigmaSlider.value())

  radiusValue.html(radiusSlider.value())
  sigmaValue.html(blurRenderer.sigma.toFixed(2))

  blurRenderer.draw(() => {
    clear()
    push()
    background(10)
    noStroke()
    lights()

    push()
    emissiveMaterial(255, 40, 40)
    translate(
      90 * sin(millis() / 500),
      60 * cos(millis() / 500),
      40 * sin(millis() / 800),
    )
    sphere(35)
    pop()

    push()
    emissiveMaterial(40, 120, 255)
    translate(
      80 * cos(millis() / 300 + 12),
      70 * sin(millis() / 600),
      60 * sin(millis() / 800 + 1),
    )
    sphere(25)
    pop()

    push()
    emissiveMaterial(255, 255, 255)
    translate(0, 110, -120)
    sphere(15)
    pop()

    push()
    fill(60)
    translate(0, 150, 0)
    rotateX(PI / 2)
    plane(900, 900)
    pop()
    pop()
  })

  if (frameCount % 10 === 0) {
    statusLine.html(
      `taps/pass: ${blurRenderer.kernel.offsets.length} ` +
        `&middot; intermediates: ${blurRenderer.getIntermediateFormatName()} ` +
        `&middot; fps: ${frameRate().toFixed(0)}`,
    )
  }
}
