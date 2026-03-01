module gpu_tensor_core_pipe3 (
  input         clk,
  input         rst,
  input         in_valid,
  input  [1:0]  in_op,
  input  [63:0] in_a,
  input  [63:0] in_b,
  input  [63:0] in_c,
  output        out_valid,
  output [63:0] out_y
);
  function [15:0] bf16_pack;
    input s;
    input [7:0] exp;
    input [6:0] frac;
    begin
      bf16_pack = {s, exp, frac};
    end
  endfunction

  function [15:0] bf16_relu;
    input [15:0] x;
    begin
      if (x[15]) bf16_relu = 16'h0000;
      else bf16_relu = x;
    end
  endfunction

  function [15:0] bf16_mul;
    input [15:0] a;
    input [15:0] b;
    reg sa;
    reg sb;
    reg sr;
    reg [7:0] ea;
    reg [7:0] eb;
    reg [7:0] er;
    reg [6:0] fa;
    reg [6:0] fb;
    reg [7:0] sig_a;
    reg [7:0] sig_b;
    reg [15:0] prod;
    reg [7:0] sig8;
    reg [6:0] frac;
    reg guard;
    reg sticky;
    reg roundb;
    begin
      sa = a[15]; sb = b[15]; ea = a[14:7]; eb = b[14:7];
      fa = a[6:0]; fb = b[6:0];
      if ((ea == 8'd0 && fa == 7'd0) || (eb == 8'd0 && fb == 7'd0)) begin
        bf16_mul = 16'h0000;
      end else begin
        sr = sa ^ sb;
        sig_a = {1'b1, fa};
        sig_b = {1'b1, fb};
        prod = sig_a * sig_b;
        er = ea + eb - 8'd127;
        if (prod[15]) begin
          sig8 = prod[15:8];
          guard = prod[7];
          sticky = |prod[6:0];
          er = er + 8'd1;
        end else begin
          sig8 = prod[14:7];
          guard = prod[6];
          sticky = |prod[5:0];
        end
        frac = sig8[6:0];
        roundb = guard & (sticky | frac[0]);
        if (roundb) begin
          if (frac == 7'h7F) begin
            frac = 7'h00;
            er = er + 8'd1;
          end else begin
            frac = frac + 7'd1;
          end
        end
        bf16_mul = bf16_pack(sr, er, frac);
      end
    end
  endfunction

  function [15:0] bf16_add;
    input [15:0] a;
    input [15:0] b;
    reg sa;
    reg sb;
    reg sr;
    reg [7:0] ea;
    reg [7:0] eb;
    reg [7:0] er;
    reg [6:0] fa;
    reg [6:0] fb;
    reg [7:0] sig_a;
    reg [7:0] sig_b;
    reg [10:0] ext_a;
    reg [10:0] ext_b;
    reg [10:0] a_al;
    reg [10:0] b_al;
    integer sh;
    integer i;
    reg sticky;
    reg [11:0] sum;
    reg [11:0] mag;
    reg [8:0] sig9;
    reg [6:0] frac;
    reg guard;
    reg roundb;
    reg stickyb;
    reg inc;
    begin
      sa = a[15]; sb = b[15]; ea = a[14:7]; eb = b[14:7];
      fa = a[6:0]; fb = b[6:0];
      if (ea == 8'd0 && fa == 7'd0) begin
        bf16_add = b;
      end else if (eb == 8'd0 && fb == 7'd0) begin
        bf16_add = a;
      end else begin
        sig_a = {1'b1, fa};
        sig_b = {1'b1, fb};
        ext_a = {sig_a, 3'b000};
        ext_b = {sig_b, 3'b000};
        er = (ea >= eb) ? ea : eb;
        a_al = ext_a;
        b_al = ext_b;

        if (er > ea) begin
          sh = er - ea;
          if (sh >= 11) begin
            sticky = |a_al;
            a_al = 11'd0;
            if (sticky) a_al[0] = 1'b1;
          end else if (sh != 0) begin
            sticky = 1'b0;
            for (i = 0; i < 11; i = i + 1) begin
              if (i < sh) sticky = sticky | a_al[i];
            end
            a_al = a_al >> sh;
            if (sticky) a_al[0] = 1'b1;
          end
        end

        if (er > eb) begin
          sh = er - eb;
          if (sh >= 11) begin
            sticky = |b_al;
            b_al = 11'd0;
            if (sticky) b_al[0] = 1'b1;
          end else if (sh != 0) begin
            sticky = 1'b0;
            for (i = 0; i < 11; i = i + 1) begin
              if (i < sh) sticky = sticky | b_al[i];
            end
            b_al = b_al >> sh;
            if (sticky) b_al[0] = 1'b1;
          end
        end

        if (sa == sb) begin
          sum = {1'b0, a_al} + {1'b0, b_al};
          sr = sa;
          mag = sum;
          if (mag[11]) begin
            sticky = mag[0];
            mag = mag >> 1;
            if (sticky) mag[0] = 1'b1;
            er = er + 8'd1;
          end
        end else begin
          if (a_al >= b_al) begin
            mag = {1'b0, a_al} - {1'b0, b_al};
            sr = sa;
          end else begin
            mag = {1'b0, b_al} - {1'b0, a_al};
            sr = sb;
          end
        end

        if (mag == 12'd0) begin
          bf16_add = 16'h0000;
        end else begin
          for (i = 0; i < 12; i = i + 1) begin
            if (mag[10] == 1'b0 && er != 8'd0) begin
              mag = mag << 1;
              er = er - 8'd1;
            end
          end

          guard = mag[2];
          roundb = mag[1];
          stickyb = mag[0];
          frac = mag[9:3];
          inc = guard & (roundb | stickyb | frac[0]);
          sig9 = {1'b0, mag[10:3]} + {8'd0, inc};
          if (sig9[8]) begin
            er = er + 8'd1;
            sig9 = sig9 >> 1;
          end
          frac = sig9[6:0];
          bf16_add = bf16_pack(sr, er, frac);
        end
      end
    end
  endfunction

  reg        v0;
  reg [1:0]  op0;
  reg [63:0] a0;
  reg [63:0] b0;
  reg [63:0] c0;

  reg        v1;
  reg [1:0]  op1;
  reg [63:0] a1;
  reg [63:0] b1;
  reg [63:0] c1;

  reg        v2;
  reg [63:0] y2;

  wire [15:0] a0_0;
  wire [15:0] a0_1;
  wire [15:0] a0_2;
  wire [15:0] a0_3;
  wire [15:0] b0_0;
  wire [15:0] b0_1;
  wire [15:0] b0_2;
  wire [15:0] b0_3;

  assign a0_0 = a0[15:0];
  assign a0_1 = a0[31:16];
  assign a0_2 = a0[47:32];
  assign a0_3 = a0[63:48];
  assign b0_0 = b0[15:0];
  assign b0_1 = b0[31:16];
  assign b0_2 = b0[47:32];
  assign b0_3 = b0[63:48];

  reg [15:0] p1_0;
  reg [15:0] p1_1;
  reg [15:0] p1_2;
  reg [15:0] p1_3;

  wire [15:0] sum01;
  wire [15:0] sum23;
  wire [15:0] sum0123;
  wire [15:0] sum_relu;
  assign sum01 = bf16_add(p1_0, p1_1);
  assign sum23 = bf16_add(p1_2, p1_3);
  assign sum0123 = bf16_add(sum01, sum23);
  assign sum_relu = bf16_relu(sum0123);

  always @(posedge clk) begin
    if (rst) begin
      v0 <= 1'b0;
      op0 <= 2'd0;
      a0 <= 64'd0;
      b0 <= 64'd0;
      c0 <= 64'd0;
    end else begin
      v0 <= in_valid;
      op0 <= in_op;
      a0 <= in_a;
      b0 <= in_b;
      c0 <= in_c;
    end
  end

  always @(posedge clk) begin
    if (rst) begin
      v1 <= 1'b0;
      op1 <= 2'd0;
      a1 <= 64'd0;
      b1 <= 64'd0;
      c1 <= 64'd0;
      p1_0 <= 16'd0;
      p1_1 <= 16'd0;
      p1_2 <= 16'd0;
      p1_3 <= 16'd0;
    end else begin
      v1 <= v0;
      op1 <= op0;
      a1 <= a0;
      b1 <= b0;
      c1 <= c0;
      if (op0 == 2'd0 || op0 == 2'd1 || op0 == 2'd3) begin
        p1_0 <= bf16_mul(a0_0, b0_0);
        p1_1 <= bf16_mul(a0_1, b0_1);
        p1_2 <= bf16_mul(a0_2, b0_2);
        p1_3 <= bf16_mul(a0_3, b0_3);
      end else begin
        p1_0 <= 16'd0;
        p1_1 <= 16'd0;
        p1_2 <= 16'd0;
        p1_3 <= 16'd0;
      end
    end
  end

  wire [15:0] c1_0;
  wire [15:0] c1_1;
  wire [15:0] c1_2;
  wire [15:0] c1_3;
  assign c1_0 = c1[15:0];
  assign c1_1 = c1[31:16];
  assign c1_2 = c1[47:32];
  assign c1_3 = c1[63:48];

  always @(posedge clk) begin
    if (rst) begin
      v2 <= 1'b0;
      y2 <= 64'd0;
    end else begin
      v2 <= v1;
      if (!v1) begin
        y2 <= 64'd0;
      end else begin
        case (op1)
          2'd0: y2 <= {p1_3, p1_2, p1_1, p1_0};
          2'd1: y2 <= {bf16_add(p1_3, c1_3), bf16_add(p1_2, c1_2), bf16_add(p1_1, c1_1), bf16_add(p1_0, c1_0)};
          2'd2: y2 <= {bf16_relu(a1[63:48]), bf16_relu(a1[47:32]), bf16_relu(a1[31:16]), bf16_relu(a1[15:0])};
          2'd3: y2 <= {48'h0, sum_relu};
          default: y2 <= 64'd0;
        endcase
      end
    end
  end

  assign out_valid = v2;
  assign out_y = y2;
endmodule