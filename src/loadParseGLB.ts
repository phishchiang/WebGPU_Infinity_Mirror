import { NodeIO } from '@gltf-transform/core';
import { getVertexLayout } from './getVertexLayout';

export async function loadAndProcessGLB(
  url: string
): Promise<{
  interleavedData: Float32Array;
  indices: Uint16Array | undefined;
  indexCount: number;
  vertexLayout: { arrayStride: number; attributes: GPUVertexAttribute[] };
}> {
  const io = new NodeIO();
  const response = await fetch(url);
  const arrayBuffer = await response.arrayBuffer();

  const document = await io.readBinary(new Uint8Array(arrayBuffer));
  const mesh = document.getRoot().listMeshes()[0]; // Assuming the first mesh is the one you want

  const primitive = mesh.listPrimitives()[0]; // Assuming the first primitive
  const positionAccessor = primitive.getAttribute('POSITION');
  const uvAccessor = primitive.getAttribute('TEXCOORD_0');
  const normalAccessor = primitive.getAttribute('NORMAL');
  const colorAccessor = primitive.getAttribute('COLOR_0');
  const indicesAccessor = primitive.getIndices();

  if (!positionAccessor) {
    throw new Error('Missing POSITION in the glTF file.');
  }

  const vertices = new Float32Array(positionAccessor!.getArray()!);
  const indices = indicesAccessor ? new Uint16Array(indicesAccessor.getArray()!) : undefined;
  const vertexNormal = normalAccessor ? new Float32Array(normalAccessor.getArray()!) : undefined;
  const uvs = uvAccessor ? new Float32Array(uvAccessor.getArray()!) : undefined;
  const colors = colorAccessor ? new Float32Array(colorAccessor.getArray()!) : undefined;

  // console.log('Original GLB vertex order:');
  // for (let i = 0; i < vertices.length / 3; i++) {
  //   const x = vertices[i * 3 + 0];
  //   const y = vertices[i * 3 + 1];
  //   const z = vertices[i * 3 + 2];
  //   console.log(`Vertex ${i}: [${x}, ${y}, ${z}]`);
  // }

  // Ensure the number of UVs matches the number of vertices
  if (uvs && uvs.length / 2 !== vertices.length / 3) {
    console.error('UV count does not match vertex count!');
    throw new Error('UV count does not match vertex count!');
  }

  const vertexLayout = new getVertexLayout({
    position: vertices.length > 0,
    normal: vertexNormal && vertexNormal.length > 0,
    color: colors && colors.length > 0,
    uv: uvs && uvs.length > 0,
  }).build();

  // Interleave positions, normals, colors, and UVs into a single array
  const interleavedData = new Float32Array((vertices.length / 3) * (vertexLayout.arrayStride / 4));
  for (let i = 0, j = 0; i < vertices.length / 3; i++) {
    interleavedData[j++] = vertices[i * 3 + 0]; // x
    interleavedData[j++] = vertices[i * 3 + 1]; // y
    interleavedData[j++] = vertices[i * 3 + 2]; // z
    if (vertexNormal && vertexNormal.length > 0) {
      interleavedData[j++] = vertexNormal[i * 3 + 0]; // nx
      interleavedData[j++] = vertexNormal[i * 3 + 1]; // ny
      interleavedData[j++] = vertexNormal[i * 3 + 2]; // nz
    }
    if (colors && colors.length > 0) {
      interleavedData[j++] = colors[i * 4 + 0]; // cr
      interleavedData[j++] = colors[i * 4 + 1]; // cg
      interleavedData[j++] = colors[i * 4 + 2]; // cb
      interleavedData[j++] = colors[i * 4 + 3]; // ca
    }
    if (uvs && uvs.length > 0) {
      interleavedData[j++] = uvs[i * 2 + 0]; // u
      interleavedData[j++] = uvs[i * 2 + 1]; // v
    }
  }

  return {
    interleavedData,
    indices,
    indexCount: indices!.length,
    vertexLayout,
  };
}