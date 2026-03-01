module gpu_regfile_64 #(parameter NREG = 32) (
  input              clk,
  input              rst,
  input      [4:0]   ra0,
  output reg [63:0]  rd0,
  input      [4:0]   ra1,
  output reg [63:0]  rd1,
  input      [4:0]   ra2,
  output reg [63:0]  rd2,
  input              we,
  input      [4:0]   wa,
  input      [63:0]  wd
);
  reg [63:0] rf [0:NREG-1];
  integer i;
  always @(posedge clk) begin
    if (rst) begin
      for (i = 0; i < NREG; i = i + 1) rf[i] <= 64'd0;
    end else begin
      if (we && (wa != 5'd0)) rf[wa] <= wd;
    end
  end
  always @(*) begin
    rd0 = (ra0 == 5'd0) ? 64'd0 : rf[ra0];
    rd1 = (ra1 == 5'd0) ? 64'd0 : rf[ra1];
    rd2 = (ra2 == 5'd0) ? 64'd0 : rf[ra2];
  end
endmodule

module gpu_exec_i16 (
  input  [1:0]  op,
  input  [63:0] a,
  input  [63:0] b,
  output [63:0] y
);
  wire signed [15:0] a0;
  wire signed [15:0] a1;
  wire signed [15:0] a2;
  wire signed [15:0] a3;
  wire signed [15:0] b0;
  wire signed [15:0] b1;
  wire signed [15:0] b2;
  wire signed [15:0] b3;

  assign a0 = a[15:0];
  assign a1 = a[31:16];
  assign a2 = a[47:32];
  assign a3 = a[63:48];

  assign b0 = b[15:0];
  assign b1 = b[31:16];
  assign b2 = b[47:32];
  assign b3 = b[63:48];

  reg [15:0] y0;
  reg [15:0] y1;
  reg [15:0] y2;
  reg [15:0] y3;

  function [15:0] relu16;
    input signed [15:0] x;
    begin
      if (x[15]) relu16 = 16'h0000;
      else relu16 = x;
    end
  endfunction

  always @(*) begin
    y0 = 16'h0; y1 = 16'h0; y2 = 16'h0; y3 = 16'h0;
    case (op)
      2'd0: begin y0 = a0 + b0; y1 = a1 + b1; y2 = a2 + b2; y3 = a3 + b3; end
      2'd1: begin y0 = a0 - b0; y1 = a1 - b1; y2 = a2 - b2; y3 = a3 - b3; end
      2'd2: begin y0 = relu16(a0); y1 = relu16(a1); y2 = relu16(a2); y3 = relu16(a3); end
      default: begin y0 = 16'h0; y1 = 16'h0; y2 = 16'h0; y3 = 16'h0; end
    endcase
  end

  assign y = {y3, y2, y1, y0};
endmodule