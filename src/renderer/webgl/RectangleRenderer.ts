/**
 * Copyright (c) 2018 The xterm.js authors. All rights reserved.
 * @license MIT
 */

import { ITerminal } from '../../Types';
import { IColorManager, IRenderDimensions } from '../Types';
import { createProgram, expandFloat32Array, PROJECTION_MATRIX } from './WebglUtils';
import { IColor } from '../Types';
import { IRenderModel, IWebGLVertexArrayObject, IWebGL2RenderingContext, ISelectionRenderModel } from './Types';
import { fill } from '../../common/TypedArrayUtils';
import { INVERTED_DEFAULT_COLOR, DEFAULT_COLOR } from '../atlas/Types';
import { is256Color } from '../atlas/CharAtlasUtils';

const enum VertexAttribLocations {
  POSITION = 0,
  SIZE = 1,
  COLOR = 2,
  UNIT_QUAD = 3
}

const vertexShaderSource = `#version 300 es
layout (location = ${VertexAttribLocations.POSITION}) in vec2 a_position;
layout (location = ${VertexAttribLocations.SIZE}) in vec2 a_size;
layout (location = ${VertexAttribLocations.COLOR}) in vec3 a_color;
layout (location = ${VertexAttribLocations.UNIT_QUAD}) in vec2 a_unitquad;

uniform mat4 u_projection;
uniform vec2 u_resolution;

out vec3 v_color;

void main() {
  vec2 zeroToOne = (a_position + (a_unitquad * a_size)) / u_resolution;
  gl_Position = u_projection * vec4(zeroToOne, 0.0, 1.0);
  v_color = a_color;
}`;

const fragmentShaderSource = `#version 300 es
precision mediump float;

in vec3 v_color;

out vec4 outColor;

void main() {
  outColor = vec4(v_color, 1);
}`;

const selectionVertexShaderSource = `#version 300 es
layout (location = ${VertexAttribLocations.POSITION}) in vec2 a_position;
layout (location = ${VertexAttribLocations.SIZE}) in vec2 a_size;

uniform mat4 u_projection;
uniform vec2 u_resolution;
uniform float u_inset;

void main() {
  vec2 pos = a_position + a_size * u_inset;
  gl_Position = u_projection * vec4(pos / u_resolution, 0.0, 1.0);
}`;

const selectionFragmentShaderSource = `#version 300 es
precision mediump float;

uniform vec3 u_color;

out vec4 outColor;

void main() {
  outColor = vec4(u_color, 1);
}`;

interface IVertices {
  attributes: Float32Array;
  count: number;
}
interface ISelectionVertices {
  vertices: Float32Array;
  outlineIndices: Uint8Array;
}

const INDICES_PER_RECTANGLE = 8;
const BYTES_PER_RECTANGLE = INDICES_PER_RECTANGLE * Float32Array.BYTES_PER_ELEMENT;

// Selection
const ATTRIBUTES_PER_SELECTION_VERTEX = 4
const BYTES_PER_SELECTION_VERTEX = ATTRIBUTES_PER_SELECTION_VERTEX * Float32Array.BYTES_PER_ELEMENT;

const INITIAL_BUFFER_RECTANGLE_CAPACITY = 20 * INDICES_PER_RECTANGLE;

const SELECTION_VERTEX_COUNT = 10;
const SELECTION_OUTLINE_INDEX_COUNT = 16;

export class RectangleRenderer {

  private _program: WebGLProgram;
  private _vertexArrayObject: IWebGLVertexArrayObject;
  private _resolutionLocation: WebGLUniformLocation;
  private _attributesBuffer: WebGLBuffer;
  private _projectionLocation: WebGLUniformLocation;
  private _bgFloat: Float32Array;
  private _selectionFloat: Float32Array;

  private _selectionProgram: WebGLProgram;
  private _selectionVao: IWebGLVertexArrayObject;
  private _selectionOutlineVao: IWebGLVertexArrayObject;
  private _selectionAttributesBuffer: WebGLBuffer;
  private _selectionResolutionLocation: WebGLUniformLocation;
  private _selectionProjectionLocation: WebGLUniformLocation;
  private _selectionColorLocation: WebGLUniformLocation;
  private _selectionInsetLocation: WebGLUniformLocation;

  private _vertices: IVertices = {
    count: 0,
    attributes: new Float32Array(INITIAL_BUFFER_RECTANGLE_CAPACITY),
  };

  private _selectionVertices: ISelectionVertices = {
    vertices: new Float32Array(SELECTION_VERTEX_COUNT * 4),
    outlineIndices: new Uint8Array(SELECTION_OUTLINE_INDEX_COUNT),
  };

  constructor(
    private _terminal: ITerminal,
    private _colorManager: IColorManager,
    private _gl: IWebGL2RenderingContext,
    private _dimensions: IRenderDimensions
  ) {
    const gl = this._gl;

    this._program = createProgram(gl, vertexShaderSource, fragmentShaderSource);
    this._selectionProgram = createProgram(gl, selectionVertexShaderSource, selectionFragmentShaderSource);

    // Uniform locations
    this._resolutionLocation = gl.getUniformLocation(this._program, 'u_resolution');
    this._projectionLocation = gl.getUniformLocation(this._program, 'u_projection');

    this._selectionResolutionLocation = gl.getUniformLocation(this._selectionProgram, 'u_resolution');
    this._selectionProjectionLocation = gl.getUniformLocation(this._selectionProgram, 'u_projection');
    this._selectionColorLocation = gl.getUniformLocation(this._selectionProgram, 'u_color');
    this._selectionInsetLocation = gl.getUniformLocation(this._selectionProgram, 'u_inset');

    // Create and set the vertex array object
    this._vertexArrayObject = gl.createVertexArray();
    gl.bindVertexArray(this._vertexArrayObject);

    // Setup a_unitquad, this defines the 4 vertices of a rectangle
    const unitQuadVertices = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]);
    const unitQuadVerticesBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, unitQuadVerticesBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, unitQuadVertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(VertexAttribLocations.UNIT_QUAD);
    gl.vertexAttribPointer(VertexAttribLocations.UNIT_QUAD, 2, this._gl.FLOAT, false, 0, 0);

    // Setup the unit quad element array buffer, this points to indices in
    // unitQuadVertuces to allow is to draw 2 triangles from the vertices
    const unitQuadElementIndices = new Uint8Array([0, 1, 3, 0, 2, 3]);
    const elementIndicesBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, elementIndicesBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, unitQuadElementIndices, gl.STATIC_DRAW);

    // Setup attributes
    this._attributesBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this._attributesBuffer);
    gl.enableVertexAttribArray(VertexAttribLocations.POSITION);
    gl.vertexAttribPointer(VertexAttribLocations.POSITION, 2, gl.FLOAT, false, BYTES_PER_RECTANGLE, 0);
    gl.vertexAttribDivisor(VertexAttribLocations.POSITION, 1);
    gl.enableVertexAttribArray(VertexAttribLocations.SIZE);
    gl.vertexAttribPointer(VertexAttribLocations.SIZE, 2, gl.FLOAT, false, BYTES_PER_RECTANGLE, 2 * Float32Array.BYTES_PER_ELEMENT);
    gl.vertexAttribDivisor(VertexAttribLocations.SIZE, 1);
    gl.enableVertexAttribArray(VertexAttribLocations.COLOR);
    gl.vertexAttribPointer(VertexAttribLocations.COLOR, 4, gl.FLOAT, false, BYTES_PER_RECTANGLE, 4 * Float32Array.BYTES_PER_ELEMENT);
    gl.vertexAttribDivisor(VertexAttribLocations.COLOR, 1);

    this._selectionVao = gl.createVertexArray();
    gl.bindVertexArray(this._selectionVao);

      //      0-----------1
      //      |           |
      // 4----2-----------3
      // |                |
      // 5-------------7--6
      // |             |
      // 8-------------9
    const selectionIndexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, selectionIndexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint8Array([
      0, 1, 2,
      2, 1, 3,
      4, 3, 5,
      5, 3, 6,
      5, 7, 8,
      8, 7, 9,
    ]), gl.STATIC_DRAW)

    this._selectionAttributesBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this._selectionAttributesBuffer);
    gl.enableVertexAttribArray(VertexAttribLocations.POSITION);
    gl.vertexAttribPointer(VertexAttribLocations.POSITION, 2, gl.FLOAT, false, BYTES_PER_SELECTION_VERTEX, 0);
    gl.enableVertexAttribArray(VertexAttribLocations.SIZE);
    gl.vertexAttribPointer(VertexAttribLocations.SIZE, 2, gl.FLOAT, false, BYTES_PER_SELECTION_VERTEX, 2 * Float32Array.BYTES_PER_ELEMENT);

    this._selectionOutlineVao = gl.createVertexArray();
    gl.bindVertexArray(this._selectionOutlineVao);

    const outlineIndexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, outlineIndexBuffer);
    gl.enableVertexAttribArray(VertexAttribLocations.POSITION);
    gl.vertexAttribPointer(VertexAttribLocations.POSITION, 2, gl.FLOAT, false, BYTES_PER_SELECTION_VERTEX, 0);
    gl.enableVertexAttribArray(VertexAttribLocations.SIZE);
    gl.vertexAttribPointer(VertexAttribLocations.SIZE, 2, gl.FLOAT, false, BYTES_PER_SELECTION_VERTEX, 2 * Float32Array.BYTES_PER_ELEMENT);

    gl.bindVertexArray(null);

    this._updateCachedColors();
  }

  public render(): void {
    const gl = this._gl;

    gl.useProgram(this._program);

    gl.bindVertexArray(this._vertexArrayObject);

    gl.uniformMatrix4fv(this._projectionLocation, false, PROJECTION_MATRIX);
    gl.uniform2f(this._resolutionLocation, gl.canvas.width, gl.canvas.height);

    // Bind attributes buffer and draw
    gl.bindBuffer(gl.ARRAY_BUFFER, this._attributesBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this._vertices.attributes, gl.DYNAMIC_DRAW);
    gl.drawElementsInstanced(this._gl.TRIANGLES, 6, gl.UNSIGNED_BYTE, 0, this._vertices.count);

    this.renderSelection();
  }

  // TODO: extract this to SelectionRenderer.ts?
  public renderSelection(): void {
    if (!this._selectionVertices) {
      return;
    }
    const gl = this._gl;
    gl.useProgram(this._selectionProgram);

    // Draw fill
    gl.bindVertexArray(this._selectionVao);

    gl.uniformMatrix4fv(this._selectionProjectionLocation, false, PROJECTION_MATRIX);
    gl.uniform2f(this._selectionResolutionLocation, gl.canvas.width, gl.canvas.height);
    gl.uniform1f(this._selectionInsetLocation, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this._selectionAttributesBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this._selectionVertices.vertices, gl.DYNAMIC_DRAW);
    gl.uniform3f(this._selectionColorLocation, 1, 1, 1);
    gl.drawElements(this._gl.TRIANGLES, 6 * 3, gl.UNSIGNED_BYTE, 0);

    // Draw outline
    gl.bindVertexArray(this._selectionOutlineVao);

    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this._selectionVertices.outlineIndices, gl.DYNAMIC_DRAW);
    gl.uniform3f(this._selectionColorLocation, 0, 0, 0);
    gl.uniform1f(this._selectionInsetLocation, 1);
    gl.drawElements(this._gl.LINES, SELECTION_OUTLINE_INDEX_COUNT, gl.UNSIGNED_BYTE, 0);

    // Unbind VAO
    gl.bindVertexArray(null);
  }

  public onResize(): void {
    this._updateViewportRectangle();
  }

  public onThemeChanged(): void {
    this._updateCachedColors();
    this._updateViewportRectangle();
  }

  private _updateCachedColors(): void {
    this._bgFloat = this._colorToFloat32Array(this._colorManager.colors.background);
    this._selectionFloat = this._colorToFloat32Array(this._colorManager.colors.selection);
  }

  private _updateViewportRectangle(): void {
    // Set first rectangle that clears the screen
    this._addRectangleFloat(
      this._vertices.attributes,
      0,
      0,
      0,
      this._terminal.cols * this._dimensions.scaledCellWidth,
      this._terminal.rows * this._dimensions.scaledCellHeight,
      this._bgFloat
    );
  }

  public updateSelection(model: ISelectionRenderModel, columnSelectMode: boolean): void {
    const terminal = this._terminal;

    if (!model.hasSelection) {
      fill(this._selectionVertices.vertices, 0, 0);
      return;
    }

    const w = this._dimensions.scaledCellWidth;
    const h = this._dimensions.scaledCellHeight;
    const startRow = model.viewportCappedStartRow;
    const endRow = model.viewportCappedEndRow;
    const startCol = model.viewportStartRow === startRow ? model.startCol : 0;
    const endCol = model.viewportEndRow === endRow ? model.endCol : terminal.cols;
    const { vertices, outlineIndices } = this._selectionVertices;
    if (columnSelectMode) {
      const v0 = { x: startCol, y: startRow };
      const v1 = { x: endCol, y: startRow };
      const v2 = { x: startCol, y: endRow + 1 };
      const v3 = { x: endCol, y: endRow + 1 };
      vertices.set([
        w * v0.x, h * v0.y, +1, +1, // v0
        w * v1.x, h * v1.y, -1, +1, // v1
        w * v2.x, h * v2.y, +1, +1, // v2
        w * v3.x, h * v3.y, -1, +1, // v3
      ]);
      fill(this._selectionVertices.vertices, 0, 4 * 4);
      outlineIndices.set([0, 1, 1, 3, 3, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    } else {
      const startRowEndCol = startRow === endRow ? model.endCol : terminal.cols;
      const middleRowsCount = Math.max(endRow - startRow - 1, 0);

      // Vertices in the format (x, y, insetX, insetY).
      //      0-----------1
      //      |           |
      // 4----2-----------3
      // |                |
      // 5-------------7--6
      // |             |
      // 8-------------9
      const v0 = { x: startCol, y: startRow };
      const v4 = { x: 0, y: v0.y + 1 };
      const v6 = { x: startRowEndCol, y: v4.y + middleRowsCount };
      const v9 = { x: endCol, y: endRow + 1 };
      vertices.set([
        w * v0.x, h * v0.y, +1, +1, // v0
        w * v6.x, h * v0.y, -1, +1, // v1
        w * v0.x, h * v4.y, +1, +1, // v2
        w * v6.x, h * v4.y, -1, +1, // v3
        w * v4.x, h * v4.y, +1, +1, // v4
        w * v4.x, h * v6.y, +1, +1, // v5
        w * v6.x, h * v6.y, -1, -1, // v6
        w * v9.x, h * v6.y, -1, -1, // v7
        w * v4.x, h * v9.y, +1, -1, // v8
        w * v9.x, h * v9.y, -1, -1, // v9
      ]);

      if (endRow - startRow === 0) {
        outlineIndices.set([0, 1, 1, 3, 3, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
      } else if (endRow - startRow === 1 && startCol > endCol) {
        outlineIndices.set([0, 1, 1, 3, 3, 2, 2, 0, 5, 7, 7, 9, 9, 8, 8, 5]);
      } else {
        outlineIndices.set([0, 1, 1, 6, 6, 7, 7, 9, 9, 8, 8, 4, 4, 2, 2, 0]);
      }
    }
  }

  public updateBackgrounds(model: IRenderModel): void {
    const terminal = this._terminal;
    const vertices = this._vertices;

    let rectangleCount = 1;

    for (let y = 0; y < terminal.rows; y++) {
      let currentStartX = -1;
      let currentBg = DEFAULT_COLOR;
      for (let x = 0; x < terminal.cols; x++) {
        const modelIndex = ((y * terminal.cols) + x) * 4;
        const bg = model.cells[modelIndex + 2];
        if (bg !== currentBg) {
          // A rectangle needs to be drawn if going from non-default to another color
          if (currentBg !== DEFAULT_COLOR) {
            const offset = rectangleCount++ * INDICES_PER_RECTANGLE;
            this._updateRectangle(vertices, offset, currentBg, currentStartX, x, y);
          }
          currentStartX = x;
          currentBg = bg;
        }
      }
      // Finish rectangle if it's still going
      if (currentBg !== DEFAULT_COLOR) {
        const offset = rectangleCount++ * INDICES_PER_RECTANGLE;
        this._updateRectangle(vertices, offset, currentBg, currentStartX, terminal.cols, y);
      }
    }
    vertices.count = rectangleCount;
  }

  private _updateRectangle(vertices: IVertices, offset: number, bg: number, startX: number, endX: number, y: number): void {
    let color: IColor | null = null;
    if (bg === INVERTED_DEFAULT_COLOR) {
      color = this._colorManager.colors.foreground;
    } else if (is256Color(bg)) {
      color = this._colorManager.colors.ansi[bg];
    }
    if (vertices.attributes.length < offset + 4) {
      vertices.attributes = expandFloat32Array(vertices.attributes, this._terminal.rows * this._terminal.cols * INDICES_PER_RECTANGLE);
    }
    const x1 = startX * this._dimensions.scaledCellWidth;
    const y1 = y * this._dimensions.scaledCellHeight;
    const r = ((color.rgba >> 24) & 0xFF) / 255;
    const g = ((color.rgba >> 16) & 0xFF) / 255;
    const b = ((color.rgba >> 8 ) & 0xFF) / 255;

    this._addRectangle(vertices.attributes, offset, x1, y1, (endX - startX) * this._dimensions.scaledCellWidth, this._dimensions.scaledCellHeight, r, g, b, 1);
  }

  private _addRectangle(array: Float32Array, offset: number, x1: number, y1: number, width: number, height: number, r: number, g: number, b: number, a: number): void {
    array[offset    ] = x1;
    array[offset + 1] = y1;
    array[offset + 2] = width;
    array[offset + 3] = height;
    array[offset + 4] = r;
    array[offset + 5] = g;
    array[offset + 6] = b;
    array[offset + 7] = a;
  }

  private _addRectangleFloat(array: Float32Array, offset: number, x1: number, y1: number, width: number, height: number, color: Float32Array): void {
    array[offset    ] = x1;
    array[offset + 1] = y1;
    array[offset + 2] = width;
    array[offset + 3] = height;
    array[offset + 4] = color[0];
    array[offset + 5] = color[1];
    array[offset + 6] = color[2];
    array[offset + 7] = color[3];
  }

  private _colorToFloat32Array(color: IColor): Float32Array {
    return new Float32Array([
      ((color.rgba >> 24) & 0xFF) / 255,
      ((color.rgba >> 16) & 0xFF) / 255,
      ((color.rgba >> 8 ) & 0xFF) / 255,
      ((color.rgba      ) & 0xFF) / 255
    ]);
  }
}
