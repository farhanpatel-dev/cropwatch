import { describe, expect, it } from "vitest";

import { boxAround, extension, formatArea, parseBounds, polygonToLatLngBounds } from "./geo.js";

describe("parseBounds", () => {
  it("accepts four numbers", () => {
    expect(parseBounds("-93.62, 41.99, -93.6, 42")).toEqual([-93.62, 41.99, -93.6, 42]);
  });
  it.each(["1,2,3", "a,b,c,d", "1,1,0,2", "-200,0,10,10", "0,-95,1,1", "1,,2,3"])(
    "rejects %s",
    (text) => expect(() => parseBounds(text)).toThrow(),
  );
});

it("extension is lower case with dot", () => {
  expect(extension("Field.JPG")).toBe(".jpg");
  expect(extension("noext")).toBe("");
});

it("polygonToLatLngBounds swaps to lat/lon", () => {
  const poly = { type: "Polygon", coordinates: [[[-93.62, 41.99], [-93.6, 41.99], [-93.6, 42], [-93.62, 42], [-93.62, 41.99]]] };
  expect(polygonToLatLngBounds(poly)).toEqual([[41.99, -93.62], [42, -93.6]]);
});

it("boxAround is roughly the requested size", () => {
  const [minLon, minLat, maxLon, maxLat] = boxAround(42, -93.6, 100).split(",").map(Number);
  expect((maxLat - minLat) * 111320).toBeCloseTo(100, 0);
  expect((maxLon - minLon) * 111320 * Math.cos((42 * Math.PI) / 180)).toBeCloseTo(100, 0);
});

it("formatArea switches units", () => {
  expect(formatArea(0.0512)).toBe("512 m²");
  expect(formatArea(3.4211)).toBe("3.42 ha");
});
