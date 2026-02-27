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
    end
  endtask

  integer k;
  reg [31:0] r0;
  reg [63:0] got;
  reg [63:0] exp;

  initial begin
    reset = 1;
    reg_req_in = 0;
    reg_ack_in = 0;
    reg_rd_wr_L_in = 1;
    reg_addr_in = 0;
    reg_data_in = 0;
    reg_src_in = 0;

    #40;
    reset = 0;

    reg_write(A(8'h01), 32'd0);
    reg_write(A(8'h02), 32'd16);
    reg_write(A(8'h03), 32'd32);
    reg_write(A(8'h04), 32'd1);

    reg_write(A(8'h10), 32'd0);
    reg_write(A(8'h11), I(4'h8, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
    reg_write(A(8'h12), 32'd1);

    reg_write(A(8'h10), 32'd1);
    reg_write(A(8'h11), I(4'h2, 3'd1, 3'd0, 3'd0, 2'd0, 16'h0000));
    reg_write(A(8'h12), 32'd1);

    reg_write(A(8'h10), 32'd2);
    reg_write(A(8'h11), I(4'h2, 3'd2, 3'd0, 3'd0, 2'd1, 16'h0000));
    reg_write(A(8'h12), 32'd1);

    reg_write(A(8'h10), 32'd3);
    reg_write(A(8'h11), I(4'h4, 3'd3, 3'd1, 3'd2, 2'd0, 16'h0000));
    reg_write(A(8'h12), 32'd1);

    reg_write(A(8'h10), 32'd4);
    reg_write(A(8'h11), I(4'h3, 3'd0, 3'd0, 3'd3, 2'd2, 16'h0000));
    reg_write(A(8'h12), 32'd1);

    reg_write(A(8'h10), 32'd5);
    reg_write(A(8'h11), I(4'h9, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
    reg_write(A(8'h12), 32'd1);

    reg_write(A(8'h10), 32'd6);
    reg_write(A(8'h11), I(4'hA, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0001));
    reg_write(A(8'h12), 32'd1);

    reg_write(A(8'h10), 32'd7);
    reg_write(A(8'h11), I(4'hF, 3'd0, 3'd0, 3'd0, 2'd0, 16'h0000));
    reg_write(A(8'h12), 32'd1);

    reg_write(A(8'h20), 32'd0);
    reg_write(A(8'h21), 32'h0002_0001);
    reg_write(A(8'h22), 32'h0004_0003);
    reg_write(A(8'h23), 32'd1);

    reg_write(A(8'h20), 32'd16);
    reg_write(A(8'h21), 32'h0009_0009);
    reg_write(A(8'h22), 32'h0009_0009);
    reg_write(A(8'h23), 32'd1);

    reg_write(A(8'h20), 32'd32);
    reg_write(A(8'h21), 32'h0);
    reg_write(A(8'h22), 32'h0);
    reg_write(A(8'h23), 32'd1);

    $display("START");
    reg_write(A(8'h00), 32'h1);

    for (k = 0; k < 200; k = k + 1) begin
      reg_read(A(8'h00), r0);
      $display("STAT busy=%0d done=%0d pc=%0d tid=%0d ir=%h",
        r0[0], r0[1],
        dut.dbg_pc, dut.dbg_tid, dut.dbg_ir
      );
      if (dut.dbg_st_we) $display("  ST addr=%0d data=%h", dut.dbg_st_addr, dut.dbg_st_data);
      if (r0[1]) begin
        reg_write(A(8'h20), 32'd32);
        reg_read(A(8'h24), r0);
        got[31:0] = r0;
        reg_read(A(8'h25), r0);
        got[63:32] = r0;
        exp = 64'h000d_000c_000b_000a;
        if (got !== exp) begin
          $display("FAIL got=%h exp=%h", got, exp);
        end else begin
          $display("PASS got=%h", got);
        end
        $finish;
      end
    end

    $display("TIMEOUT");
    $finish;
  end

endmodule