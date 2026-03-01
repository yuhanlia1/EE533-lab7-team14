module gpu_imem #(parameter DEPTH = 256, parameter AW = 8) (
  input               clk,
  input  [AW-1:0]      raddr,
  output reg [31:0]    rdata,
  input               we,
  input  [AW-1:0]      waddr,
  input  [31:0]        wdata
);
  reg [31:0] mem [0:DEPTH-1];
  integer i;
  initial begin
    for (i = 0; i < DEPTH; i = i + 1) mem[i] = 32'h0;
    rdata = 32'h0;
  end
  always @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    rdata <= mem[raddr];
  end
endmodule