module gpu_dmem #(parameter DEPTH = 1024, parameter AW = 10) (
  input               clk,
  input               we,
  input  [AW-1:0]      waddr,
  input  [63:0]        wdata,
  input  [AW-1:0]      raddr,
  output reg [63:0]    rdata,
  input  [AW-1:0]      dbg_raddr,
  output reg [63:0]    dbg_rdata
);
  reg [63:0] mem [0:DEPTH-1];
  integer j;
  initial begin
    for (j = 0; j < DEPTH; j = j + 1) mem[j] = 64'h0;
    rdata = 64'h0;
    dbg_rdata = 64'h0;
  end
  always @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    rdata <= mem[raddr];
    dbg_rdata <= mem[dbg_raddr];
  end
endmodule