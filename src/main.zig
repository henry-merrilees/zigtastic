const std = @import("std");

const or_gate = [_][3]f32{
    // {input, input, output}
    [_]f32{ 0, 0, 0 },
    [_]f32{ 0, 1, 1 },
    [_]f32{ 1, 0, 1 },
    [_]f32{ 1, 1, 1 },
};

const and_gate = [_][3]f32{
    // {input, input, output}
    [_]f32{ 0, 0, 0 },
    [_]f32{ 0, 1, 0 },
    [_]f32{ 1, 0, 0 },
    [_]f32{ 1, 1, 1 },
};

const xor_gate = [_][3]f32{
    // {input, input, output}
    [_]f32{ 0, 0, 0 },
    [_]f32{ 0, 1, 1 },
    [_]f32{ 1, 0, 1 },
    [_]f32{ 1, 1, 0 },
};

const always_1 = [_][3]f32{
    // {input, input, output}
    [_]f32{ 0, 0, 1 },
    [_]f32{ 0, 1, 1 },
    [_]f32{ 1, 0, 1 },
    [_]f32{ 1, 1, 1 },
};

const times_2 = [_][3]f32{
    // {input, input, output}
    [_]f32{ 0, 0, 0 },
    [_]f32{ 1, 0, 2 },
    [_]f32{ 2, 0, 4 },
    [_]f32{ 3, 0, 6 },
};

const train = and_gate;

var rand = std.rand.DefaultPrng.init(10);
fn rand_float() f32 {
    return rand.random().float(f32);
}

fn rand_float_range(min: f32, max: f32) f32 {
    return rand_float() * (max - min) + min;
}

fn rand_floats(comptime len: usize) [len]f32 {
    var result: [len]f32 = undefined;
    for (result) |r| {
        r.* = rand_float();
    }
    return result;
}

fn rand_floats_range(comptime len: usize, min: f32, max: f32) [len]f32 {
    var result: [len]f32 = undefined;
    for (&result) |*r| {
        r.* = rand_float_range(min, max);
    }
    return result;
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + std.math.exp(-x));
}

fn forward(comptime W: usize, ws: [W]f32, b: f32, xs: [W]f32) f32 {
    var y: f32 = 0.0;
    for (xs, ws) |x, w| {
        y += x * w;
    }
    y += b;
    return sigmoid(y);
}

fn cost(comptime W: usize, ws: [W]f32, b: f32) f32 {
    var result: f32 = 0.0;
    for (train) |row| {
        const xs: [W]f32 = row[0..W].*;
        const y = row[W];
        var y_pred = forward(W, ws, b, xs);
        var err = y_pred - y;
        result += std.math.pow(f32, err, 2);
    }
    result /= @as(f32, train.len);

    return result;
}

pub fn main() !void {
    const W = 2;
    var ws = rand_floats_range(W, -5.0, 5.0);
    var b = rand_float() * 10.0 - 5.0;

    var eps: f32 = 1e-4;
    var lr: f32 = 1e-3;
    // epoch 1 through 5
    for (1..40000) |epoch| {
        var c = cost(W, ws, b);
        std.debug.print("{} c: {}\n", .{ epoch, c });
        std.debug.print("\t\t\tw: ({}, {}), b: {}\n", .{ ws[0], ws[1], b });

        // calculate an orthogonal step of size eps in each dimension
        //  ws*I
        var ortho_steps = [_][W]f32{ws} ** W;
        // ... += eps*I
        for (&ortho_steps, 0..) |*s, i| {
            s.*[i] += eps;
        }

        var dws: [W]f32 = undefined;
        for (ortho_steps, &dws) |os, *dw| {
            dw.* = (cost(W, os, b) - c) / eps;
        }

        var db = (cost(W, ws, b + eps) - c) / eps;
        // update weights
        for (&ws, dws) |*w, dw| {
            w.* -= dw * lr;
        }
        b -= db * lr;
    }

    var cc = cost(W, ws, b);
    std.debug.print("cost: {}\n", .{cc});

    // final predictions
    for (train) |row| {
        var xs = row[0..W].*; // swap in W
        var y = row[W];
        var y_pred = forward(W, ws, b, xs);

        std.debug.print("x: ", .{});
        for (xs) |x| {
            std.debug.print("{} ", .{x});
        }
        std.debug.print("y_expected: {}, y_pred: {} ", .{ y, y_pred });
        std.debug.print("error: {}\n", .{y_pred - y});
    }

    std.debug.print("final ws: ({}, {})\n", .{ ws[0], ws[1] });
    std.debug.print("final b: {}\n", .{b});
}
