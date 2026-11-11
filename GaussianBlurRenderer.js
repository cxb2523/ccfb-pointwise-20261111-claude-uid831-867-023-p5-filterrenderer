// Hard cap on the kernel half-width in pixels. A radius of 64 means 129 taps per
// pass (258 per pixel across both passes) at full resolution, which is already
// heavy. If this ever needs to grow, the next step should be rendering into
// half-resolution framebuffers and adding an upsample pass rather than just
// raising the cap: downsampling shrinks the kernel work quadratically, while a
// wider full-resolution kernel only makes the two blur passes more expensive,
// and an upsample pass is needed to return to the canvas resolution without
// changing the blur's apparent size.
const MAX_RADIUS = 64

// Used when no radius is provided at construction.
const DEFAULT_RADIUS = 10

// Sigma values below this produce a kernel that is effectively a point sample
// (and zero/negative/NaN sigma blows up the weight normalization below), so we
// clamp to it instead.
const MIN_SIGMA = 0.5

class GaussianBlurRenderer extends Renderer {
  constructor(target, options = {}) {
    // Base class creates this.fbo (the scene target, which keeps its depth
    // buffer for 3D drawing) and the shader. The two ping-pong intermediates
    // are created here once their format is known.
    super(target, options)

    this.radius = this.normalizeRadius(
      options.radius === undefined ? DEFAULT_RADIUS : options.radius,
    )
    this._userSigma = options.sigma
    this.sigma = this.normalizeSigma(options.sigma)

    this.intermediateFormat = this.detectIntermediateFormat()
    // fboA/fboB are the ping-pong targets: the horizontal pass reads one and
    // writes the other, then the vertical pass reads that result and writes
    // back into the first one. They only hold a full-screen shaded quad, so
    // they need no depth, stencil, or MSAA samples. NEAREST filtering both
    // matches the kernel's integer-pixel taps and avoids WebGL1 float textures
    // becoming incomplete without OES_texture_float_linear, which would sample
    // as black.
    const intermediateOptions = {
      ...options,
      format: this.intermediateFormat,
      depth: false,
      stencil: false,
      antialias: false,
      textureFiltering: this.target.NEAREST,
    }
    this.fboA = target.createFramebuffer(intermediateOptions)
    this.fboB = target.createFramebuffer(intermediateOptions)
    this.kernel = { offsets: [], weights: [] }
    this.rebuildKernel()
  }

  vert() {
    return GaussianBlurRenderer.vert
  }

  frag() {
    return GaussianBlurRenderer.frag
  }

  normalizeRadius(radius) {
    if (!Number.isFinite(radius) || radius < 0) return 0
    return Math.min(Math.round(radius), MAX_RADIUS)
  }

  normalizeSigma(sigma) {
    // Fall back to a conventional sigma derived from the radius (one third of
    // it, roughly matching the old baked-in s = 0.5/3 relationship) when unset
    // or invalid. Never let it reach zero: the Gaussian would divide by it and
    // normalization would collapse into a divide-by-zero.
    if (!Number.isFinite(sigma) || sigma <= 0) {
      return Math.max(this.radius / 3, MIN_SIGMA)
    }
    return Math.max(sigma, MIN_SIGMA)
  }

  // Float intermediates keep precision across the two blur passes. An unsigned
  // byte target quantizes after the horizontal pass, so sampling it again in
  // the vertical pass can show banding on smooth gradients and glow.
  // Returns a p5 format constant (the strings p5.Framebuffer expects), not a
  // raw GL enum: createFramebuffer rejects numeric GL formats as unknown.
  detectIntermediateFormat() {
    const gl = this.target._renderer.GL
    if (this.target.webglVersion === WEBGL2) {
      // p5 itself gates float rendering on both extensions: color-buffer
      // float makes RGBA32F renderable, and float-blend is needed because the
      // renderer keeps blending enabled.
      if (
        !gl.getExtension('EXT_color_buffer_float') ||
        !gl.getExtension('EXT_float_blend')
      ) {
        return this.target.UNSIGNED_BYTE
      }
    } else {
      // WebGL1 gates float rendering behind two separate extensions: one for
      // float textures at all, one for attaching them to a framebuffer.
      if (
        !gl.getExtension('OES_texture_float') ||
        !gl.getExtension('WEBGL_color_buffer_float')
      ) {
        return this.target.UNSIGNED_BYTE
      }
    }
    // Extensions alone don't guarantee a complete framebuffer on every driver,
    // so verify renderability with a throwaway float target before committing.
    return this.floatFramebufferIsRenderable(gl)
      ? this.target.FLOAT
      : this.target.UNSIGNED_BYTE
  }

  floatFramebufferIsRenderable(gl) {
    const texture = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    const internalFormat =
      this.target.webglVersion === WEBGL2 ? gl.RGBA32F : gl.RGBA
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      internalFormat,
      4,
      4,
      0,
      gl.RGBA,
      gl.FLOAT,
      null,
    )

    const framebuffer = gl.createFramebuffer()
    const previous = gl.getParameter(gl.FRAMEBUFFER_BINDING)
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer)
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      texture,
      0,
    )
    const renderable =
      gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE

    // Restore p5's framebuffer binding and clean up the probe objects.
    gl.bindFramebuffer(gl.FRAMEBUFFER, previous)
    gl.deleteFramebuffer(framebuffer)
    gl.deleteTexture(texture)
    return renderable
  }

  // Human-readable name of the intermediate texture format, useful for
  // displaying which precision path (float or byte fallback) is active.
  getIntermediateFormatName() {
    return this.intermediateFormat === this.target.FLOAT
      ? 'FLOAT'
      : 'UNSIGNED_BYTE'
  }

  setRadius(radius) {
    this.radius = this.normalizeRadius(radius)
    // An explicitly-set sigma is preserved; otherwise the derived default
    // tracks the new radius.
    this.sigma =
      this._userSigma === undefined
        ? this.normalizeSigma(undefined)
        : this.normalizeSigma(this._userSigma)
    this.rebuildKernel()
  }

  setSigma(sigma) {
    this._userSigma = sigma
    this.sigma = this.normalizeSigma(sigma)
    this.rebuildKernel()
  }

  // Build the 1D Gaussian on the CPU from the current sigma. The shader only
  // knows how to walk the uniform arrays, so changing radius/sigma never
  // requires recompiling a shader and there is no fixed-length kernel baked
  // into GLSL.
  rebuildKernel() {
    const radius = this.radius
    const sigma = this.sigma
    if (radius === 0) {
      // Identity kernel: the same shader blits the scene untouched, while the
      // two intermediate render targets never receive a draw.
      this.kernel = { offsets: [0], weights: [1] }
      return
    }
    const offsets = []
    const weights = []
    let sum = 0
    for (let i = -radius; i <= radius; i++) {
      const weight = Math.exp(-(i * i) / (2 * sigma * sigma))
      offsets.push(i)
      weights.push(weight)
      sum += weight
    }
    // Renormalize over the truncated window so a finite radius does not darken
    // the image. normalizeSigma guarantees a positive, finite denominator.
    for (let i = 0; i < weights.length; i++) {
      weights[i] /= sum
    }
    this.kernel = { offsets, weights }
  }

  getUniforms(texture, direction) {
    // Framebuffer storage includes pixel density, so one texel in UV space is
    // 1 / (css size * density).
    const texelWidth = 1 / (this.fboA.width * this.fboA.density)
    const texelHeight = 1 / (this.fboA.height * this.fboA.density)
    return {
      uImg: texture,
      uTexelSize: [texelWidth, texelHeight],
      uDirection: direction, // 0 = horizontal, 1 = vertical
      uKernelSize: this.kernel.weights.length,
      uOffsets: this.kernel.offsets,
      uWeights: this.kernel.weights,
    }
  }

  blit(texture, direction) {
    this.target.noStroke()
    this.target.shader(this.shader)
    const uniforms = this.getUniforms(texture, direction)
    for (const key in uniforms) {
      this.shader.setUniform(key, uniforms[key])
    }
    this.target.plane(this.target.width, this.target.height)
  }

  draw(cb) {
    // Render the scene into the source framebuffer first.
    this.fbo.draw(() => {
      this.target.push()
      cb()
      this.target.pop()
    })

    this.target.push()

    // Zero radius: skip both blur render-target passes entirely and blit the
    // scene straight to the canvas with the identity kernel, so passthrough
    // pays no extra framebuffer draws.
    if (this.radius === 0) {
      this.blit(this.fbo.color, 0)
      this.target.pop()
      return
    }

    // Horizontal pass: scene fbo -> fboA
    this.fboA.draw(() => {
      this.target.push()
      this.target.clear()
      this.blit(this.fbo.color, 0)
      this.target.pop()
    })

    // Vertical pass: fboA -> fboB
    this.fboB.draw(() => {
      this.target.push()
      this.target.clear()
      this.blit(this.fboA.color, 1)
      this.target.pop()
    })

    // Present the ping-pong result on the main canvas. WEBGL coordinates place
    // the image's top-left at (x, y), so negative half-dimensions fill it.
    // resetShader first: otherwise image() sees this shader's uImg sampler,
    // treats it as a texture shader (isTextureShader only checks for samplers),
    // and re-runs the stale vertical-pass uniforms instead of a plain blit.
    this.target.resetShader()
    this.target.image(
      this.fboB,
      -this.target.width / 2,
      -this.target.height / 2,
    )
    this.target.pop()
  }

  remove() {
    super.remove()
    this.fboA.remove()
    this.fboB.remove()
  }
}

p5.prototype.createGaussianBlurRenderer = function (options) {
  return new GaussianBlurRenderer(this, options)
}

GaussianBlurRenderer.vert = `
precision highp float;

attribute vec3 aPosition;
attribute vec2 aTexCoord;

uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;

varying highp vec2 vVertTexCoord;

void main(void) {
  vec4 positionVec4 = vec4(aPosition, 1.0);
  gl_Position = uProjectionMatrix * uModelViewMatrix * positionVec4;
  vVertTexCoord = aTexCoord;
}
`

GaussianBlurRenderer.frag = `
precision highp float;
varying highp vec2 vVertTexCoord;

uniform sampler2D uImg;
uniform vec2 uTexelSize;
uniform int uDirection;
uniform int uKernelSize;
// Upper bound is 2 * MAX_RADIUS + 1 = 129 taps.
const int MAX_KERNEL_SIZE = 129;
uniform float uOffsets[MAX_KERNEL_SIZE];
uniform float uWeights[MAX_KERNEL_SIZE];

void main() {
  vec4 color = vec4(0.0);
  vec2 axis = uDirection == 0 ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  for (int i = 0; i < MAX_KERNEL_SIZE; i++) {
    if (i >= uKernelSize) break;
    vec2 sampleCoord =
      vVertTexCoord + axis * uOffsets[i] * uTexelSize;
    color += texture2D(uImg, sampleCoord) * uWeights[i];
  }
  gl_FragColor = color;
}
`
