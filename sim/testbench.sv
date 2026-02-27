`timescale 1ns/1ps

`define UDP_REG_ADDR_WIDTH 23
`define CPCI_NF2_DATA_WIDTH 32

module tb;
  reg clk;
  reg reset;

  reg  reg_req_in;
  reg  reg_ack_in;
  reg  reg_rd_wr_L_in;
  reg  [`UDP_REG_ADDR_WIDTH-1:0]  reg_addr_in;
  reg  [`CPCI_NF2_DATA_WIDTH-1:0] reg_data_in;
  reg  [1:0] reg_src_in;

  wire reg_req_out;
  wire reg_ack_out;
  wire reg_rd_wr_L_out;
  wire [`UDP_REG_ADDR_WIDTH-1:0]  reg_addr_out;
  wire [`CPCI_NF2_DATA_WIDTH-1:0] reg_data_out;
  wire [1:0] reg_src_out;

  localparam GPU_ADDR_PREFIX = 8'h7F;

  function [`UDP_REG_ADDR_WIDTH-1:0] A;
    input [7:0] off;
    begin
      A = {GPU_ADDR_PREFIX, 7'h00, off};
    end
  endfunction

  function [31:0] I;
    input [3:0] op;
    input [2:0] rd;
    input [2:0] rs1;
    input [2:0] rs2;
    input [1:0] bsel;
    input [15:0] imm16;
    begin
      I = {op, rd, rs1, rs2, bsel, 1'b0, imm16};
    end
  endfunction

  gpu_datapath #(
    .UDP_REG_SRC_WIDTH(2),
    .IMEM_DEPTH(256),
    .DMEM_DEPTH(1024),
    .IMEM_AW(8),
    .DMEM_AW(10),
    .GPU_ADDR_PREFIX(GPU_ADDR_PREFIX)
  ) dut (
    .clk(clk),
    .reset(reset),
    .reg_req_in(reg_req_in),
    .reg_ack_in(reg_ack_in),
    .reg_rd_wr_L_in(reg_rd_wr_L_in),
    .reg_addr_in(reg_addr_in),
    .reg_data_in(reg_data_in),
    .reg_src_in(reg_src_in),
    .reg_req_out(reg_req_out),
    .reg_ack_out(reg_ack_out),
    .reg_rd_wr_L_out(reg_rd_wr_L_out),
    .reg_addr_out(reg_addr_out),
    .reg_data_out(reg_data_out),
    .reg_src_out(reg_src_out)
  );

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  task reg_write;
    input [`UDP_REG_ADDR_WIDTH-1:0] addr;
    input [31:0] data;
    begin
      @(posedge clk);
      reg_req_in <= 1'b1;
      reg_ack_in <= 1'b0;
      reg_rd_wr_L_in <= 1'b0;
      reg_addr_in <= addr;
      reg_data_in <= data;
      reg_src_in <= 2'd0;
      @(posedge clk);
      while (reg_ack_out == 1'b0) @(posedge clk);
      reg_req_in <= 1'b0;
      @(posedge clk);
    end
  endtask

  task reg_read;
    input  [`UDP_REG_ADDR_WIDTH-1:0] addr;
    output [31:0] data;
    begin
      @(posedge clk);
      reg_req_in <= 1'b1;
      reg_ack_in <= 1'b0;
      reg_rd_wr_L_in <= 1'b1;
      reg_addr_in <= addr;
      reg_data_in <= 32'h0;
      reg_src_in <= 2'd0;
      @(posedge clk);
      while (reg_ack_out == 1'b0) @(posedge clk);
      data = reg_data_out;
      reg_req_in <= 1'b0;
      @(posedge clk);
    end
  endtask

  task imem_wr;
    input [7:0] addr;
    input [31:0] data;
    begin
      reg_write(A(8'h10), {24'h0, addr});
      reg_write(A(8'h11), data);
      reg_write(A(8'h12), 32'h1);
    end
  endtask

  task dmem_wr64;
    input [31:0] addr;
    input [63:0] data;
    begin
      reg_write(A(8'h20), addr);
      reg_write(A(8'h21), data[31:0]);
      reg_write(A(8'h22), data[63:32]);
      reg_write(A(8'h23), 32'h1);
    end
  endtask

  task dmem_rd64;
    input [31:0] addr;
    output [63:0] data;
    reg [31:0] lo;
    reg [31:0] hi;
    begin
      reg_write(A(8'h20), addr);
      reg_read(A(8'h24), lo);
      reg_read(A(8'h25), hi);
      data = {hi, lo};
    end
  endtask

  task soft_reset;
    begin
      reset = 1'b1;
      repeat(6) @(posedge clk);
      reset = 1'b0;
      repeat(4) @(posedge clk);
    end
  endtask

  task set_cfg;
    input [31:0] abase;
    input [31:0] bbase;
    input [31:0] cbase;
    input [31:0] n;
    begin
      reg_write(A(8'h01), abase);
      reg_write(A(8'h02), bbase);
      reg_write(A(8'h03), cbase);
      reg_write(A(8'h04), n);
    end
  endtask

  task start_core;
    begin
      reg_write(A(8'h00), 32'h1);
    end
  endtask

  task wait_done;
    input integer max_iters;
    reg [31:0] st;
    integer k;
    begin
      for (k = 0; k < max_iters; k = k + 1) begin
        reg_read(A(8'h00), st);
        $display("STAT k=%0d busy=%0d done=%0d pc=%0d tid=%0d ir=%h", k, st[0], st[1], dut.dbg_pc, dut.dbg_tid, dut.dbg_ir);
        if (dut.dbg_st_we) $display("  ST addr=%0d data=%h", dut.dbg_st_addr, dut.dbg_st_data);
        if (st[1]) begin
          k = max_iters;
        end
      end
    end
  endtask

  task load_prog_vadd;
    begin
      imem_wr(8'd0, I(4'h8, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd1, I(4'h2, 3'd1, 3'd0, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd2, I(4'h2, 3'd2, 3'd0, 3'd0, 2'd1, 16'h0000));
      imem_wr(8'd3, I(4'h4, 3'd3, 3'd1, 3'd2, 2'd0, 16'h0000));
      imem_wr(8'd4, I(4'h3, 3'd0, 3'd0, 3'd3, 2'd2, 16'h0000));
      imem_wr(8'd5, I(4'h9, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd6, I(4'hA, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0001));
      imem_wr(8'd7, I(4'hF, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
    end
  endtask

  task load_prog_vsub;
    begin
      imem_wr(8'd0, I(4'h8, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd1, I(4'h2, 3'd1, 3'd0, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd2, I(4'h2, 3'd2, 3'd0, 3'd0, 2'd1, 16'h0000));
      imem_wr(8'd3, I(4'h5, 3'd3, 3'd1, 3'd2, 2'd0, 16'h0000));
      imem_wr(8'd4, I(4'h3, 3'd0, 3'd0, 3'd3, 2'd2, 16'h0000));
      imem_wr(8'd5, I(4'h9, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd6, I(4'hA, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0001));
      imem_wr(8'd7, I(4'hF, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
    end
  endtask

  task load_prog_relu_i16;
    begin
      imem_wr(8'd0, I(4'h8, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd1, I(4'h2, 3'd1, 3'd0, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd2, I(4'h7, 3'd3, 3'd1, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd3, I(4'h3, 3'd0, 3'd0, 3'd3, 2'd2, 16'h0000));
      imem_wr(8'd4, I(4'h9, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd5, I(4'hA, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0001));
      imem_wr(8'd6, I(4'hF, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
    end
  endtask

  task load_prog_bf16_mul;
    begin
      imem_wr(8'd0, I(4'h8, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd1, I(4'h2, 3'd1, 3'd0, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd2, I(4'h2, 3'd2, 3'd0, 3'd0, 2'd1, 16'h0000));
      imem_wr(8'd3, I(4'hB, 3'd3, 3'd1, 3'd2, 2'd0, 16'h0000));
      imem_wr(8'd4, I(4'h3, 3'd0, 3'd0, 3'd3, 2'd2, 16'h0000));
      imem_wr(8'd5, I(4'h9, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd6, I(4'hA, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0001));
      imem_wr(8'd7, I(4'hF, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
    end
  endtask

  task load_prog_bf16_fma;
    begin
      imem_wr(8'd0, I(4'h8, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd1, I(4'h2, 3'd1, 3'd0, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd2, I(4'h2, 3'd2, 3'd0, 3'd0, 2'd1, 16'h0000));
      imem_wr(8'd3, I(4'h2, 3'd4, 3'd0, 3'd0, 2'd2, 16'h0000));
      imem_wr(8'd4, I(4'hC, 3'd3, 3'd1, 3'd2, 2'd0, 16'h0004));
      imem_wr(8'd5, I(4'h3, 3'd0, 3'd0, 3'd3, 2'd2, 16'h0000));
      imem_wr(8'd6, I(4'h9, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
      imem_wr(8'd7, I(4'hA, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0001));
      imem_wr(8'd8, I(4'hF, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
    end
  endtask

  task check64;
    input [63:0] got;
    input [63:0] exp;
    input [127:0] tag;
    begin
      if (got !== exp) begin
        $display("FAIL %0s got=%h exp=%h", tag, got, exp);
        $finish;
      end else begin
        $display("PASS %0s got=%h", tag, got);
      end
    end
  endtask

  reg [63:0] x;

  initial begin
    reset = 1;
    reg_req_in = 0;
    reg_ack_in = 0;
    reg_rd_wr_L_in = 1;
    reg_addr_in = 0;
    reg_data_in = 0;
    reg_src_in = 0;

    soft_reset;

    $display("\n=== TEST1: i16 vadd (2 threads) ===");
    set_cfg(32'd0, 32'd16, 32'd32, 32'd2);
    load_prog_vadd;
    dmem_wr64(32'd0,  64'h0004_0003_0002_0001);
    dmem_wr64(32'd1,  64'h0008_0007_0006_0005);
    dmem_wr64(32'd16, 64'h0009_0009_0009_0009);
    dmem_wr64(32'd17, 64'h0004_0003_0002_0001);
    dmem_wr64(32'd32, 64'h0);
    dmem_wr64(32'd33, 64'h0);
    start_core;
    wait_done(200);
    dmem_rd64(32'd32, x);
    check64(x, 64'h000D_000C_000B_000A, "vadd.t0");
    dmem_rd64(32'd33, x);
    check64(x, 64'h000C_000A_0008_0006, "vadd.t1");

    $display("\n=== TEST2: i16 vsub (2 threads) ===");
    soft_reset;
    set_cfg(32'd0, 32'd16, 32'd32, 32'd2);
    load_prog_vsub;
    dmem_wr64(32'd0,  64'h0004_0003_0002_0001);
    dmem_wr64(32'd1,  64'h0008_0007_0006_0005);
    dmem_wr64(32'd16, 64'h0001_0001_0001_0001);
    dmem_wr64(32'd17, 64'h0002_0003_0004_0005);
    dmem_wr64(32'd32, 64'h0);
    dmem_wr64(32'd33, 64'h0);
    start_core;
    wait_done(200);
    dmem_rd64(32'd32, x);
    check64(x, 64'h0003_0002_0001_0000, "vsub.t0");
    dmem_rd64(32'd33, x);
    check64(x, 64'h0006_0004_0002_0000, "vsub.t1");

    $display("\n=== TEST3: i16 relu (2 threads) ===");
    soft_reset;
    set_cfg(32'd0, 32'd16, 32'd32, 32'd2);
    load_prog_relu_i16;
    dmem_wr64(32'd0,  64'h0004_FFFD_0002_FFFF);
    dmem_wr64(32'd1,  64'h0008_0007_FFFA_FFFB);
    dmem_wr64(32'd32, 64'h0);
    dmem_wr64(32'd33, 64'h0);
    start_core;
    wait_done(200);
    dmem_rd64(32'd32, x);
    check64(x, 64'h0004_0000_0002_0000, "relu_i16.t0");
    dmem_rd64(32'd33, x);
    check64(x, 64'h0008_0007_0000_0000, "relu_i16.t1");

    $display("\n=== TEST4: bf16 vmul (2 threads) ===");
    soft_reset;
    set_cfg(32'd0, 32'd16, 32'd32, 32'd2);
    load_prog_bf16_mul;
    dmem_wr64(32'd0,  64'h4080_4040_4000_3F80);
    dmem_wr64(32'd1,  64'h4100_3F00_C000_BF80);
    dmem_wr64(32'd16, 64'hBF80_3F80_3F00_4000);
    dmem_wr64(32'd17, 64'h3E80_4000_4000_BF80);
    dmem_wr64(32'd32, 64'h0);
    dmem_wr64(32'd33, 64'h0);
    start_core;
    wait_done(400);
    dmem_rd64(32'd32, x);
    check64(x, 64'hC080_4040_3F80_4000, "bf16_mul.t0");
    dmem_rd64(32'd33, x);
    check64(x, 64'h4000_3F80_C080_3F80, "bf16_mul.t1");

    $display("\n=== TEST5: bf16 vfma (2 threads) ===");
    soft_reset;
    set_cfg(32'd0, 32'd16, 32'd32, 32'd2);
    load_prog_bf16_fma;
    dmem_wr64(32'd0,  64'h4080_4040_4000_3F80);
    dmem_wr64(32'd1,  64'h4100_3F00_C000_BF80);
    dmem_wr64(32'd16, 64'hBF80_3F80_3F00_4000);
    dmem_wr64(32'd17, 64'h3E80_4000_4000_BF80);
    dmem_wr64(32'd32, 64'h3F80_3F80_3F80_3F80);
    dmem_wr64(32'd33, 64'h3F80_3F80_3F80_3F80);
    start_core;
    wait_done(600);
    dmem_rd64(32'd32, x);
    check64(x, 64'hC040_4080_4000_4040, "bf16_fma.t0");
    dmem_rd64(32'd33, x);
    check64(x, 64'h4040_4000_C040_4000, "bf16_fma.t1");

    $display("\nALL TESTS PASS");
    $finish;
  end

endmodule
